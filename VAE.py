# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import os
import time
import random
import pickle
import numpy as np

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from PIL import Image  # Sustituto de scipy.misc.imresize
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *

from net import *  # Asegúrate de que este archivo está disponible

im_size = 128


def loss_function(recon_x, x, mu, logvar):
    # Reconstrucción (MSE)
    BCE = torch.mean((recon_x - x)**2)

    # KLD tal y como se define en el paper de VAE
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def process_batch(batch):
    """
    Reemplaza la función que usaba scipy.misc.imresize por Pillow (PIL).
    """
    data = []
    for x in batch:
        # x debería ser un numpy array con forma (H, W, C), típicamente uint8
        img = Image.fromarray(x)  
        # Redimensionar con interpolación bilineal
        img = img.resize((im_size, im_size), Image.BILINEAR)
        # Convertir a numpy array, quedará (H, W, C)
        arr = np.array(img)
        # Transponer a (C, H, W) para PyTorch
        arr = arr.transpose((2, 0, 1))
        data.append(arr)

    # Crear tensor en GPU, normalizar a [-1, 1]
    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def main():
    batch_size = 128
    z_size = 512

    # Instancia del VAE (definido en net.py)
    vae = VAE(zsize=z_size, layer_count=5)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005
    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    train_epoch = 40

    # Vector de ruido para generar imágenes
    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1).cuda()

    for epoch in range(train_epoch):
        vae.train()

        # Carga de datos preprocesados (asegúrate de que existan data_fold_0.pkl, etc.)
        with open('data_fold_%d.pkl' % (epoch % 5), 'rb') as pkl:
            data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))
        random.shuffle(data_train)

        # batch_provider construye lotes y aplica process_batch
        batches = batch_provider(data_train, batch_size, process_batch, report_progress=True)

        rec_loss = 0.0
        kl_loss = 0.0

        epoch_start_time = time.time()

        # Ajuste de LR cada 8 épocas
        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("Learning rate reduced a la época:", epoch + 1)

        i = 0
        for x in batches:
            vae.train()
            vae.zero_grad()

            # Forward: salida y parámetros latentes
            rec, mu, logvar = vae(x)

            # Cálculo de pérdidas
            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()

            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            i += 1
            # Guardar y mostrar info cada 60 iteraciones
            if i % 60 == 0:
                rec_loss /= 60
                kl_loss /= 60
                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time

                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                
                # Reset de acumuladores
                rec_loss = 0.0
                kl_loss = 0.0

                # Modo evaluación: generar imágenes de ejemplo
                with torch.no_grad():
                    vae.eval()
                    # Reconstrucción
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()

                    os.makedirs('results_rec', exist_ok=True)
                    save_image(
                        resultsample.view(-1, 3, im_size, im_size),
                        'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png'
                    )

                    # Generar imágenes nuevas
                    x_gen = vae.decode(sample1)
                    resultsample = x_gen * 0.5 + 0.5
                    resultsample = resultsample.cpu()

                    os.makedirs('results_gen', exist_ok=True)
                    save_image(
                        resultsample.view(-1, 3, im_size, im_size),
                        'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png'
                    )

        del batches
        del data_train

    print("Training finished! Saving model...")
    torch.save(vae.state_dict(), "VAEmodel.pkl")


if __name__ == '__main__':
    main()
