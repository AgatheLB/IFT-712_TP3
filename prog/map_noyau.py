# -*- coding: utf-8 -*-

#####
# Philippe Spino (spip2401), Gabriel Gibeau Sanchez (gibg2501), Agathe Le Bouler (leba3207)
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        self.x_train = x_train
        K = self.get_kernel(x_train)
        print(K.shape)
        self.a = np.linalg.inv(K + self.lamb * np.identity(len(x_train))).dot(t_train)

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champ ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        K = self.get_kernel(x)
        y = np.sum(K.T.dot(self.a))
        return int(y > 0.5)

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t - prediction) ** 2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        space = 15
        sigmas = np.linspace(1e-9, 2, num=space)
        lambdas = np.linspace(1e-9, 2, num=space)
        C = np.linspace(0, 5, num=space)
        M = np.linspace(2, 6, num=space)
        D = np.linspace(1e-5, 1e-2, num=space)

        self.x_train = x_tab
        K = self.get_kernel(x_tab)

        # On cherche la combinaison donnant le meilleure score.
        best_score = -np.inf
        best_combo = None
        if self.noyau == "rbf":
            for s in sigmas:
                for l in lambdas:
                    self.sigma_square = s
                    self.lamb = l
                    K = self.get_kernel(x_tab)
                    a = np.linalg.inv(K + l * np.identity(len(x_tab))).dot(t_tab)
                    score = np.sum(K.T.dot(a))
                    if score > best_score:
                        best_score = score
                        self.a = a
                        best_combo = (s,l)
            print('la meilleur combinaison etait: sigma_square:',best_combo[0], ', lambda: ',best_combo[1])
        elif self.noyau == "lineaire":
            for lamb in lambdas:
                self.lamb = lamb
                K = self.get_kernel(x_tab)
                a = np.linalg.inv(K + 1 * np.identity(len(x_tab))).dot(t_tab)
                score = np.sum(K.T.dot(a))
                if score > best_score:
                    best_score = score
                    best_combo = lamb
                    self.a = a
            print('le meilleur lamda etait: ', best_combo)
        elif self.noyau == "polynomial":
            for m in M:
                for c in C:
                    for lamb in lambdas:
                        self.M = m
                        self.c = c
                        self.lamb = lamb
                        K = self.get_kernel(x_tab)
                        a = np.linalg.inv(K + lamb * np.identity(len(x_tab))).dot(t_tab)
                        score = np.sum(K.T.dot(a))
                        if score > best_score:
                            best_score = score
                            self.a = a
                            best_combo = (m,c)
            print('la meilleur combinaison etait: m:', best_combo[0], ', c:', best_combo[1])
        elif self.noyau == "sigmoidal":
            for d in D:
                for lamb in lambdas:
                    self.d = d
                    self.lamb = lamb
                    K = self.get_kernel(x_tab)
                    a = np.linalg.inv(K + lamb * np.identity(len(x_tab))).dot(t_tab)
                    score = np.sum(K.T.dot(a))
                    if score > best_score:
                        best_score = score
                        best_combo = (d,lamb)
                        self.a = a
            print('la meilleur combinaison etait: d:', best_combo[0], ', lamb: ', best_combo[1])
        else:
            raise ValueError("Noyau non supporté.")



    def affichage(self, x_tab, t_tab):
        
        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()

    def get_kernel(self, x):
        if self.noyau == "rbf":
            # TODO: fix K size (currently dims:(N,1))
            K = np.exp(-np.linalg.norm(self.x_train - x, axis=1) / 2 * self.sigma_square)
        elif self.noyau == "lineaire":
            K = self.x_train.dot(x.T)
        elif self.noyau == "polynomial":
            K = (self.x_train.dot(x.T) + self.c) ** self.M
        elif self.noyau == "sigmoidal":
            K = np.tanh(self.b * self.x_train.dot(x.T) + self.d)
        else:
            raise ValueError("Noyau non supporté.")
        return K  # size NxN
