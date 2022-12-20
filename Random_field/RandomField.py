import numpy as np
import os
import math
import cmath

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):

      def call(self, inputs, **kwargs):

        """
         This function uses the estimation of the eigenvalues as the input
         and computes the auto-covariance matrix of the field.
         input : Eigenvalues (Tensor, size = (batch_size, latent_size), dtype = Float)
         output : Batch covariance matrix (Tensor, size = (batch_size*latent_dim, latent_di), dtype = complex

         """

        eigenvalues = inputs  # the eigenvalues of the field size (batch_size x lat_dim)
        lat_dim = eigenvalues.get_shape()[1]
        batch_size = eigenvalues.get_shape()[0]

        batch_covariance = np.zeros(((batch_size * lat_dim), lat_dim), dtype=np.complex_)

        DFT_mat = np.zeros((lat_dim, lat_dim), dtype=np.complex_)
        two_phi_i = 2 * math.pi * (complex(0, 1) / complex(lat_dim, 0))

        for j in range(0, lat_dim):
            for k in range(0, lat_dim):
                aa = cmath.exp(((two_phi_i) * j) * k)
                DFT_mat[j][k] = complex(round(aa.real), round(aa.imag))
        # DFT_mat = [[cmath.exp((2 * math.pi*(complex(0, 1)) / complex(lat_dim, 0)) * j * k) for j in range(lat_dim -
        # 1)] for k in range(lat_dim - 1)]

        for i in range(batch_size):
            sample_eign = eigenvalues[i, :]
            iden_matrix = tf.zeros((tf.shape(sample_eign)[0], tf.shape(sample_eign)[0]), dtype=float)
            sigma_matrix = tf.linalg.set_diag(iden_matrix, sample_eign)
            # DFT matrix for the size of (latent_size x latent_size)

            DFT_Mat_inv = np.linalg.inv(DFT_mat)
            # # compute the covariance matrix B = F-1*D*F
            embedded_cov = DFT_mat * sigma_matrix * DFT_Mat_inv
            batch_covariance[i:i + lat_dim, :] = embedded_cov

            # generate the sample from CN(0, 2I)
            a_epsilon = tf.keras.backend.random_normal(shape=(lat_dim, batch_size))
            b_epsilon = tf.keras.backend.random_normal(shape=(lat_dim, batch_size))

            zeta = np.zeros((lat_dim, batch_size), dtype=np.complex_)
            zeta.real = a_epsilon
            zeta.imag = b_epsilon

            #  Y (latent_size, batch_size)
            #  DFT(latent_size, latent_size),
            # sigma(latent_size, latent_size),
            # zeta (latent_size, batch_size),
            q = DFT_mat * (tf.linalg.sqrtm(sigma_matrix))
            Y = (np.dot(q, zeta)).real.transpose()

        return batch_covariance, Y


# if __name__ == '__main__':
#     a = [[1, 2, 3, 4], [3, 4, 3, 5], [3, 4, 3, 5]]
#     eigenvalues = tf.convert_to_tensor(a, dtype=float)
#     batch_covariance, Y = RF(eigenvalues)
#     print(batch_covariance)
