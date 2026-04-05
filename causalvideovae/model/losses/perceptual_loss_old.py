import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips
from einops import rearrange

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return torch.pow((x-y), 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        super(NLayerDiscriminator3D, self).__init__()
        norm_layer = nn.GroupNorm if use_actnorm else nn.BatchNorm3d
        
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult, 8) if use_actnorm else norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult, 8) if use_actnorm else norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=0)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class LPIPSWithDiscriminator3D(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        learn_logvar: bool = False,
        wavelet_weight=0.01,
        distill_weight=1.0,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.distill_weight = distill_weight
        self.perceptual_loss = lpips.LPIPS(net="alex").eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e5).detach()
        d_weight = d_weight 
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        consistence,
        consistence_weight,
        split="train",
        weights=None,
        last_layer=None,
        wavelet_coeffs=None,
        student_latents=None,
        teacher_latents=None
    ):
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0: # Generator
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            reconstructions = rearrange(
                reconstructions, "b c t h w -> (b t) c h w"
            ).contiguous()
            rec_loss = self.loss_func(inputs, reconstructions) #l1 loss
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss = rec_loss + self.perceptual_weight * p_loss 
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = (
                torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            )
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            if posteriors is not None:
                kl_loss = posteriors.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if wavelet_coeffs:
                wl_loss_l2 = torch.sum(l1(wavelet_coeffs[0], wavelet_coeffs[1])) / bs
                wl_loss_l3 = torch.sum(l1(wavelet_coeffs[2], wavelet_coeffs[3])) / bs
                wl_loss = wl_loss_l2 + wl_loss_l3
            else:
                wl_loss = torch.tensor(0.0)

            if teacher_latents is not None:
                assert student_latents is not None
                distill_loss = torch.sum(l1(teacher_latents, student_latents)) / bs
            else:
                distill_loss = torch.tensor(0.0)
            
            inputs = rearrange(inputs, "(b t) c h w -> b c t h w", t=t).contiguous()
            reconstructions = rearrange(
                reconstructions, "(b t) c h w -> b c t h w", t=t
            ).contiguous()
 
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            # 关键修复：移除了 .detach()
            consistence_loss_weighted = consistence_weight * consistence.mean()
            
            # 确保所有损失都包含在最终损失中
            total_loss = weighted_nll_loss + consistence_loss_weighted
            
            # 添加KL损失
            if posteriors is not None:
                total_loss = total_loss + self.kl_weight * kl_loss
                
            # 添加小波损失
            if wavelet_coeffs is not None:
                total_loss = total_loss + self.wavelet_weight * wl_loss
                
            # 添加蒸馏损失
            if teacher_latents is not None:
                total_loss = total_loss + self.distill_weight * distill_loss
            
            loss = total_loss
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): weighted_nll_loss.detach().mean(),
                "{}/consistence_loss".format(split): consistence.detach().mean(),
                "{}/consistence_loss_weighted".format(split): consistence_loss_weighted.detach().mean(),
            }
            
            # 添加其他损失的日志
            if posteriors is not None:
                log["{}/kl_loss".format(split)] = kl_loss.detach().mean()
            if wavelet_coeffs is not None:
                log["{}/wavelet_loss".format(split)] = wl_loss.detach().mean()
            if teacher_latents is not None:
                log["{}/distill_loss".format(split)] = distill_loss.detach().mean()
            
            return loss, log
        elif optimizer_idx == 1: # Discriminator
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class VQLPIPSWithDiscriminator3D(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        codebook_weight=1.0,
        # --- Discriminator Loss ---
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        learn_logvar: bool = False,
        wavelet_weight=0.01,
        distill_weight=1.0,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.distill_weight = distill_weight
        self.perceptual_loss = lpips.LPIPS(net="alex").eval()
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e5).detach()
        d_weight = d_weight 
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        consistence,
        optimizer_idx,
        global_step,
        consistence_weight,
        split="train",
        weights=None,
        last_layer=None,
        wavelet_coeffs=None,
        student_latents=None,
        teacher_latents=None,
        codebook_loss=None,
    ):
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0: # Generator
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            reconstructions = rearrange(
                reconstructions, "b c t h w -> (b t) c h w"
            ).contiguous()
            rec_loss = self.loss_func(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = (
                torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            )
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            if wavelet_coeffs:
                wl_loss_l2 = torch.sum(l1(wavelet_coeffs[0], wavelet_coeffs[1])) / bs
                wl_loss_l3 = torch.sum(l1(wavelet_coeffs[2], wavelet_coeffs[3])) / bs
                wl_loss = wl_loss_l2 + wl_loss_l3
            else:
                wl_loss = torch.tensor(0.0)

            if teacher_latents is not None:
                assert student_latents is not None
                distill_loss = torch.sum(l1(teacher_latents, student_latents)) / bs
            else:
                distill_loss = torch.tensor(0.0)
            
            inputs = rearrange(inputs, "(b t) c h w -> b c t h w", t=t).contiguous()
            reconstructions = rearrange(
                reconstructions, "(b t) c h w -> b c t h w", t=t
            ).contiguous()
 
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, device=reconstructions.device, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            
            loss = (
                weighted_nll_loss
                + self.kl_weight * 0  # VQ doesn't use KL
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss
                + self.wavelet_weight * wl_loss
                + self.distill_weight * distill_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/codebook_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): weighted_nll_loss.detach().mean(),
                "{}/wl_loss".format(split): wl_loss.detach().mean(),
                "{}/distill_loss".format(split): distill_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            return loss, log
        elif optimizer_idx == 1: # Discriminator
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log