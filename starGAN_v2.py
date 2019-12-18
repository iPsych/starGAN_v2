import os
import matplotlib.pyplot as plt
import torch
import glob
from torch import nn
import numpy as np
from torch import autograd
import torch.nn.functional as F

from utils import get_scheduler, weights_init
from model import Generator, StyleEncoder, MappingNetwork, Discriminator
from custom._utils_torch import reset_gradients, reshape_batch_torch, show_batch_torch
from custom._visualizer import preprocess, show, clear_jupyter_console


class StarGAN(nn.Module):
    def __init__(self, config, train_loader, test_loader):
        super(StarGAN, self).__init__()

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_source, self.test_domain, _ = next(iter(self.test_loader))
        self.test_source = self.test_source.to(self.device)
        self.test_domain = self.test_domain.view(-1, 1, 1).to(self.device)
        self.test_batch_size, _, self.height, self.width = self.test_source.size()
        self.save_img_cnt = 0
        self.loss = {}
        self.repo = {}

        self.iter_size = len(self.train_loader)
        self.epoch_size = config['max_iter'] // self.iter_size + 1

        lr = config['lr']
        lr_F = config['lr_F']
        beta1 = config['beta1']
        beta2 = config['beta2']
        init = config['init']
        # weight_decay = config['weight_decay']

        self.batch_size = config['batch_size']
        self.gan_type = config['gan_type']
        self.max_iter = config['max_iter']

        self.path_sample = os.path.join('./results/samples', config['save_name'])
        self.path_model = os.path.join('./results/models', config['save_name'])

        self.w_style = config['w_style']
        self.w_ds = config['w_ds']
        self.w_cyc = config['w_cyc']
        self.w_regul = config['w_regul']

        self.num_domain = config['num_domain']
        self.dim_style = config['dim_style']
        self.dim_latent = config['mapping_network']['dim_latent']

        self.generator = Generator(config['gen'])  # 29072960
        self.mapping_network = MappingNetwork(config['mapping_network'], self.num_domain, self.dim_style)
        self.style_encoder = StyleEncoder(config['style_encoder'], self.num_domain, self.dim_style)
        self.discriminator = Discriminator(config['dis'], self.num_domain, 1)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr, (beta1, beta2))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr, (beta1, beta2))
        self.optimizer_style = torch.optim.Adam(self.style_encoder.parameters(), lr, (beta1, beta2))
        self.optimizer_F = torch.optim.Adam(self.mapping_network.parameters(), lr_F, (beta1, beta2))

        # self.scheduler_g = get_scheduler(self.optimizer_g, config['celebA'])
        # self.scheduler_d = get_scheduler(self.optimizer_d, config['celebA'])

        self.apply(weights_init(init))

        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()

        self.to(self.device)

    # def update_scheduler(self):
    #     if self.current_epoch >= 10 and self.scheduler_d and self.scheduler_g:
    #         self.scheduler_d.step()
    #         self.scheduler_g.step()

    def calc_adversarial_loss(self, logit, is_real):
        if self.gan_type == 'bce':
            target_fn = torch.ones_like if is_real else torch.zeros_like
            loss = self.criterion_bce(logit, target_fn(logit))

        elif self.gan_type == 'lsgan':
            target_fn = torch.ones_like if is_real else torch.zeros_like
            loss = self.criterion_l2(logit, target_fn(logit))

        elif self.gan_type == 'wgan':
            if is_real:
                loss = - torch.mean(logit)
            else:
                loss = torch.mean(logit)
        else:
            raise NotImplementedError("Unsupported gan type: {}".format(self.gan_type))

        return loss

    def calc_r1(self, real_images, logit_real):
        batch_size = real_images.size(0)
        grad_dout = autograd.grad(
            outputs=logit_real.sum(), inputs=real_images,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == real_images.size())
        reg = grad_dout2.view(batch_size, -1).sum(1).mean()
        return reg

    def calc_gp(self, real_images, fake_images):
        raise DeprecationWarning("")
        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device)
        interpolated = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
        prob_interpolated, _ = self.discriminator(interpolated)

        grad_outputs = torch.ones(prob_interpolated.size()).to(self.device)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def assign_variables(self, real_image):
        self.real = real_image.to(self.device)
        random_noise = torch.randn(real_image.size(0), 16).to(self.device)

        self.style_map = self.mapping_network(random_noise)
        self.style_enc = self.style_encoder(self.real)

    def generate_random_nosie(self):
        random_noise = torch.randn(self.batch_size, 16).to(self.device)
        random_domain = torch.randint(self.num_domain, (self.batch_size, 1, 1)).to(self.device)
        return random_noise, random_domain

    def eval_mode_all(self):
        self.discriminator.eval()
        self.generator.eval()

    def update_d(self, real, real_domain, random_noise, random_domain):
        reset_gradients([self.optimizer_g, self.optimizer_d])
        real.requires_grad_()  # TODO :

        style_mapped = self.mapping_network(random_noise, random_domain)
        fake = self.generator(real, style_mapped)

        # Adv
        logit_real = self.discriminator(real, real_domain)
        logit_fake = self.discriminator(fake.detach(), random_domain)

        adv_d_real = self.calc_adversarial_loss(logit_real, is_real=True)
        adv_d_fake = self.calc_adversarial_loss(logit_fake, is_real=False)

        if self.config['gan_type'] == 'bce':
            regul = self.calc_r1(real, logit_real)
        elif self.config['gan_type'] == 'wgan':
            regul = self.calc_gp(real, fake)

        loss_d = adv_d_fake + adv_d_real + regul
        loss_d.backward()
        self.optimizer_d.step()

        self.loss['adv_d_fake'] = adv_d_fake.item()
        self.loss['adv_d_real'] = adv_d_real.item()
        self.loss['regul'] = regul.item()

        self.logit_real = logit_real
        self.logit_fake_d = logit_fake

    def update_g(self, real, real_domain, random_noise, random_domain):
        reset_gradients([self.optimizer_g, self.optimizer_d])

        style_mapped = self.mapping_network(random_noise, random_domain)
        style_origin = self.style_encoder(real, real_domain)
        fake = self.generator(real, style_mapped)
        style_recon = self.style_encoder(fake, random_domain)
        image_recon = self.generator(fake, style_origin)

        # Adversarial
        logit_fake = self.discriminator(fake, random_domain)
        adv_g = self.calc_adversarial_loss(logit_fake, is_real=True)

        # Style recon
        style_recon_loss = self.criterion_l1(style_mapped, style_recon) * self.w_style

        # Style diversification
        random_noise1 = torch.randn(self.batch_size, self.dim_latent).to(self.device)
        random_noise2 = torch.randn(self.batch_size, self.dim_latent).to(self.device)  # TODO : 재활용 할까?
        random_domain1 = torch.randint(self.num_domain, (self.batch_size, 1, 1)).to(self.device)

        s1 = self.mapping_network(random_noise1, random_domain1)
        s2 = self.mapping_network(random_noise2, random_domain1)
        fake1 = self.generator(real, s1)
        fake2 = self.generator(real, s2)

        ds_loss = - self.criterion_l1(fake1, fake2) * self.w_ds

        # Cycle consistency
        cyc_loss = self.criterion_l1(real, image_recon) * self.w_cyc

        loss_g = adv_g + style_recon_loss + ds_loss + cyc_loss
        loss_g.backward()
        self.optimizer_g.step()

        self.loss['adv_g'] = adv_g.item()
        self.loss['style_recon_loss'] = style_recon_loss.item()
        self.loss['ds_loss'] = ds_loss.item()
        self.loss['cyc_loss'] = cyc_loss.item()

        # TODO : refactoring
        self.real = real
        self.real_domain = real_domain
        self.random_noise = random_noise
        self.random_domain = random_domain

        self.random_noise1 = random_noise1
        self.random_noise2 = random_noise2
        self.random_domain1 = random_domain1

        self.logit_fake_g = logit_fake

        self.style_mapped = style_mapped
        self.style_origin = style_origin
        self.fake = fake
        self.image_recon = image_recon
        self.style_recon = style_recon

    def train_starGAN(self, init_epoch):
        d_step, g_step = self.config['d_step'], self.config['g_step']
        log_iter = self.config['log_iter']
        image_display_iter = self.config['image_display_iter']
        image_save_iter = self.config['image_save_iter']

        for epoch in range(init_epoch, self.epoch_size):
            self.current_epoch = epoch
            self.save_img_cnt = 0
            for iters, (real, real_domain, _) in enumerate(self.train_loader):
                # self.update_scheduler()

                # real, real_domain = real.to(self.device), real_domain.view(-1, 1, 1).to(self.device)
                real, real_domain = real.to(self.device), real_domain.to(self.device)
                random_noise, random_domain = self.generate_random_nosie()

                if not iters & d_step:
                    self.update_d(real, real_domain, random_noise, random_domain)

                if not iters % g_step:
                    self.update_g(real, real_domain, random_noise, random_domain)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                if not (iters + 1) % log_iter:
                    self.print_log(epoch, iters)

                    if not (iters + 1) % image_display_iter:
                        show_batch_torch(torch.cat([self.real, self.fake.clamp(-1, 1)]), n_rows=2, n_cols=-1)

                        if not (iters + 1) % image_save_iter:
                            self.test_sample = self.generate_test_samples(save=True)
                            clear_jupyter_console()

    def print_log(self, epoch, iters):
        adv_d_real = self.loss['adv_d_real']
        adv_d_fake = self.loss['adv_d_fake']
        regul = self.loss['regul']
        adv_g = self.loss['adv_g']
        style_recon_loss = self.loss['style_recon_loss']
        ds_loss = self.loss['ds_loss']
        cyc_loss = self.loss['cyc_loss']

        print(
            "[Epoch {}/{}, iters: {}/{}] " \
            "- Adv: {:5.4} {:5.4} / {:5.4}, Style recon: {:5.4}, DS: {:5.4}, Cyc : {:5.4}, Regul : {:5.4}".format(
                epoch, self.epoch_size, iters + 1, self.iter_size,
                adv_d_real, adv_d_fake, adv_g, style_recon_loss, ds_loss, cyc_loss, regul
            )
        )

    def save_models(self, epoch):
        os.makedirs(self.path_model, exist_ok=True)

        state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            # 'scheduler_d': self.scheduler_d.state_dict(),  # TODO
            # 'scheduler_g': self.scheduler_g.state_dict(),
            'current_epoch': epoch,
        }

        save_name = os.path.join(self.path_model, "epoch_{:02}".format(epoch))
        torch.save(state, save_name)

    def load_models(self, epoch=False):
        if not epoch:
            last_model_path = sorted(glob.glob(os.path.join(self.path_model, '*')))[-1]
            epoch = int(last_model_path.split('/')[-1].split('_')[1][:2])

        save_name = os.path.join(self.path_model, "epoch_{:02}".format(epoch))
        checkpoint = torch.load(save_name)

        # weight
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        # self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        # self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.current_epoch = checkpoint['current_epoch']

    def resume_train(self, restart_epoch=False):
        self.load_models(restart_epoch)
        print("Resume Training - Epoch: ", restart_epoch)
        self.train_starGAN(restart_epoch + 1)

    def generate_test_samples(self, save):
        os.makedirs(self.path_sample, exist_ok=True)

        # self.eval_mode_all()
        with torch.no_grad():
            reference, reference_domain, _ = next(iter(self.test_loader))
            reference, reference_domain = reference.to(self.device), reference_domain.to(self.device)

            style_reference = self.style_encoder(reference, reference_domain)
            style_reference = style_reference.repeat(1, reference.size(0), 1).view(-1, 1, self.dim_style)
            source = self.test_source.repeat(reference.size(0), 1, 1, 1).view(-1, 3, self.height, self.width)
            generated = self.generator(source, style_reference).clamp(-1, 1)

            right_concat, _, _ = reshape_batch_torch(
                torch.cat([self.test_source, generated]), n_cols=self.test_batch_size, n_rows=-1
            )

            left_concat = torch.cat([torch.zeros_like(reference[:1]), reference])
            left_concat, _, _ = reshape_batch_torch(left_concat, n_cols=1, n_rows=-1)

            save_image = preprocess(np.concatenate([left_concat, right_concat], axis=1))

            if save:
                save_name = os.path.join(self.path_sample,
                                         "{:02}_{:02}.jpg".format(self.current_epoch, self.save_img_cnt))
                self.save_img_cnt += 1
                plt.imsave(save_name, save_image)
                print("Test samples Saved:" + save_name)
        return save_image


if __name__ == '__main__':
    from utils import get_config
    from data import get_data_loader

    config = './config/celeba_HQ.yaml'
    if True:
        dataset_root = '/mnt/disks/sdb/datasets'
    else:
        dataset_root = '/Users/bochan/_datasets'

    config = get_config(config)
    train_loader = get_data_loader(config, dataset_root, is_train=True)
    test_loader = get_data_loader(config, dataset_root, is_train=False)
    self = StarGAN(config, train_loader, test_loader)

    d_step, g_step = self.config['d_step'], self.config['g_step']
    log_iter = self.config['log_iter']
    image_display_iter = self.config['image_display_iter']
    image_save_iter = self.config['image_save_iter']

    real, real_domain, _ = next(iter(self.train_loader))
    real, real_domain = real.to(self.device), real_domain.view(-1, 1, 1).to(self.device)
    random_noise, random_domain = self.generate_random_nosie()

    # --- update_d
    reset_gradients([self.optimizer_g, self.optimizer_d])
    style_mapped = self.mapping_network(random_noise, random_domain)
    fake = self.generator(real, style_mapped)

    # Adv
    logit_real = self.discriminator(real, real_domain)
    logit_fake = self.discriminator(fake.detach(), random_domain)

    adv_d_real = self.calc_adversarial_loss(logit_real, is_real=True)
    adv_d_fake = self.calc_adversarial_loss(logit_fake, is_real=False)

    print(style_mapped.shape, torch.max(style_mapped), torch.min(style_mapped))
    print(fake.shape)
    show_batch_torch(fake)
