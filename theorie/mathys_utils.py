import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff   # à mettre en haut du fichier, avec les autres imports


def loss(theta1, theta2, a=0.1, b=10.0):
    """Calcule la loss L(theta) = a*theta1^2 + b*theta2^2.

    inputs :
        theta1 (float ou np.ndarray) : première coordonnée de theta (direction plate)
        theta2 (float ou np.ndarray) : deuxième coordonnée de theta (direction raide)
        a      (float)               : coefficient devant theta1^2
        b      (float)               : coefficient devant theta2^2
    output :
        L (float ou np.ndarray) : valeur de la loss (même forme que theta1 et theta2)
    """
    L = a * theta1 * theta1 + b * theta2 * theta2
    return L


def plot_loss_surface(a=0.1, b=10.0, lim=3.0, n=600):
    """Trace deux coupes 2D de L(theta1, theta2) = a*theta1^2 + b*theta2^2 :
    à gauche L(theta1, 0) (direction plate), à droite L(0, theta2) (direction raide).
    Les deux graphiques ont les mêmes axes et la même échelle en x et en y.

    inputs :
        a   (float) : coefficient devant theta1^2
        b   (float) : coefficient devant theta2^2
        lim (float) : theta1 et theta2 sont tracés dans [-lim, lim]
        n   (int)   : nombre de points par courbe
    output :
        fig (go.Figure) : figure Plotly avec les deux coupes
    """
    t = np.linspace(-lim, lim, n)
    ymax = 2 * lim

    # valeurs de la loss le long de chaque axe
    L1 = loss(t, 0.0, a, b)    # theta2 = 0
    L2 = loss(0.0, t, a, b)    # theta1 = 0

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=(f"L(θ₁, 0) = {a:g}·θ₁²  (direction plate)",
                        f"L(0, θ₂) = {b:g}·θ₂²  (direction raide)"),
    )

    # courbe de gauche : direction plate
    fig.add_trace(go.Scatter(
        x=t, y=L1, mode="lines",
        line=dict(color="royalblue", width=3), name="L(θ₁, 0)",
        hovertemplate="θ₁ = %{x:.2f}<br>L = %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # courbe de droite : direction raide
    fig.add_trace(go.Scatter(
        x=t, y=L2, mode="lines",
        line=dict(color="crimson", width=3), name="L(0, θ₂)",
        hovertemplate="θ₂ = %{x:.2f}<br>L = %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    # point du minimum (0, 0) sur chaque graphique
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=10, color="black"),
        name="minimum", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=10, color="black"),
        name="minimum", showlegend=False,
    ), row=1, col=2)

    # mêmes axes pour les deux graphiques, et même échelle en x et en y
    fig.update_xaxes(range=[-lim, lim], dtick=1, zeroline=True)
    fig.update_yaxes(range=[0, ymax], dtick=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="θ₁", row=1, col=1)
    fig.update_xaxes(title_text="θ₂", row=1, col=2)
    fig.update_yaxes(title_text="L(θ)", row=1, col=1)
    fig.update_yaxes(title_text="L(θ)", row=1, col=2)
    fig.update_layout(
        title=f"Coupes de L = {a:g}·θ₁² + {b:g}·θ₂²  (mêmes axes, échelle 1:1)",
        width=1000, height=600,
    )
    return fig


def gradient(theta1, theta2, a=0.1, b=10.0):
    """Calcule le gradient de L = a*theta1^2 + b*theta2^2.

    inputs :
        theta1 (float ou np.ndarray) : première coordonnée de theta
        theta2 (float ou np.ndarray) : deuxième coordonnée de theta
        a      (float)               : coefficient devant theta1^2
        b      (float)               : coefficient devant theta2^2
    output :
        g1 (float ou np.ndarray) : dérivée partielle dL/dtheta1 = 2*a*theta1
        g2 (float ou np.ndarray) : dérivée partielle dL/dtheta2 = 2*b*theta2
    """
    g1 = 2 * a * theta1
    g2 = 2 * b * theta2
    return g1, g2


def plot_gradient_surface(a=0.1, b=10.0, lim=3.0, lim2=0.3, n=300):
    """Trace deux coupes 2D du gradient de L = a*theta1^2 + b*theta2^2 :
    à gauche dL/dtheta1 en fonction de theta1, à droite dL/dtheta2 en fonction de theta2.
    Les deux graphiques partagent le même axe vertical pour comparer les pentes.

    inputs :
        a    (float) : coefficient devant theta1^2
        b    (float) : coefficient devant theta2^2
        lim  (float) : theta1 est tracé dans [-lim, lim]
        lim2 (float) : theta2 est tracé dans [-lim2, lim2]
        n    (int)   : nombre de points par courbe
    output :
        fig (go.Figure) : figure Plotly avec les deux dérivées partielles
    """
    t1 = np.linspace(-lim, lim, n)
    t2 = np.linspace(-lim2, lim2, n)

    g1, g_inutile = gradient(t1, 0.0, a, b)    # on garde seulement dL/dtheta1
    g_inutile, g2 = gradient(0.0, t2, a, b)    # on garde seulement dL/dtheta2

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(f"∂L/∂θ₁ = {2*a:g}·θ₁  (direction plate)",
                        f"∂L/∂θ₂ = {2*b:g}·θ₂  (direction raide)"),
    )
    fig.add_trace(go.Scatter(
        x=t1, y=g1, mode="lines", line=dict(color="royalblue", width=3), name="∂L/∂θ₁",
        hovertemplate="θ₁ = %{x:.2f}<br>∂L/∂θ₁ = %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t2, y=g2, mode="lines", line=dict(color="crimson", width=3), name="∂L/∂θ₂",
        hovertemplate="θ₂ = %{x:.3f}<br>∂L/∂θ₂ = %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    # au minimum, le gradient est nul
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=10, color="black"),
        name="minimum : ∇L = 0", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers", marker=dict(size=10, color="black"),
        name="minimum : ∇L = 0", showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="θ₁", row=1, col=1)
    fig.update_xaxes(title_text="θ₂", row=1, col=2)
    fig.update_yaxes(title_text="dérivée partielle", row=1, col=1)
    fig.update_layout(
        title=f"Gradient de L = {a:g}·θ₁² + {b:g}·θ₂² le long de chaque axe",
        width=1000, height=450,
    )
    return fig


def hessian(a=0.1, b=10.0):
    """Calcule la Hessienne de L = a*theta1^2 + b*theta2^2 (elle est constante),
    ainsi que ses valeurs propres et vecteurs propres.

    inputs :
        a (float) : coefficient devant theta1^2
        b (float) : coefficient devant theta2^2
    output :
        H       (np.ndarray, shape (2, 2)) : matrice Hessienne [[2a, 0], [0, 2b]]
        eigvals (np.ndarray, shape (2,))   : valeurs propres, rangées par ordre croissant
        eigvecs (np.ndarray, shape (2, 2)) : vecteurs propres (un par colonne)
    """
    H = np.array([[2 * a, 0.0],
                  [0.0, 2 * b]])

    # H est symétrique : eigh renvoie les valeurs propres dans l'ordre croissant
    eigvals, eigvecs = np.linalg.eigh(H)
    return H, eigvals, eigvecs


def plot_hessian_surface(a=0.1, b=10.0, lim=3.0, lim2=0.3, n=300):
    """Trace deux coupes 2D des dérivées secondes de L = a*theta1^2 + b*theta2^2 :
    à gauche d2L/dtheta1^2 = 2a, à droite d2L/dtheta2^2 = 2b.
    Pour une loss quadratique ce sont deux droites horizontales : la courbure
    ne dépend pas de theta.

    inputs :
        a    (float) : coefficient devant theta1^2
        b    (float) : coefficient devant theta2^2
        lim  (float) : theta1 est tracé dans [-lim, lim]
        lim2 (float) : theta2 est tracé dans [-lim2, lim2]
        n    (int)   : nombre de points par courbe
    output :
        fig (go.Figure) : figure Plotly avec les deux dérivées secondes
    """
    t1 = np.linspace(-lim, lim, n)
    t2 = np.linspace(-lim2, lim2, n)
    H, eigvals, eigvecs = hessian(a, b)

    h11 = H[0, 0]          # d2L/dtheta1^2
    h22 = H[1, 1]          # d2L/dtheta2^2

    # la dérivée seconde est la même pour tous les theta : tableaux constants
    y1 = h11 * np.ones(n)
    y2 = h22 * np.ones(n)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(f"∂²L/∂θ₁² = {h11:g}  (direction plate)",
                        f"∂²L/∂θ₂² = {h22:g}  (direction raide)"),
    )
    fig.add_trace(go.Scatter(
        x=t1, y=y1, mode="lines", line=dict(color="royalblue", width=3),
        name="∂²L/∂θ₁²", hovertemplate="θ₁ = %{x:.2f}<br>∂²L/∂θ₁² = %{y:g}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t2, y=y2, mode="lines", line=dict(color="crimson", width=3),
        name="∂²L/∂θ₂²", hovertemplate="θ₂ = %{x:.3f}<br>∂²L/∂θ₂² = %{y:g}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="θ₁", row=1, col=1)
    fig.update_xaxes(title_text="θ₂", row=1, col=2)
    fig.update_yaxes(title_text="courbure", range=[0, 1.1 * max(h11, h22)], row=1, col=1)
    fig.update_layout(
        title=f"Dérivées secondes de L = {a:g}·θ₁² + {b:g}·θ₂² : constantes partout",
        width=1000, height=450,
    )
    return fig


def plot_sensitivity(a=0.1, b=10.0, eps=0.1, n=361):
    """Trace la sensibilité de la loss selon la direction, en partant du minimum theta* = (0, 0).

    On fait un pas de longueur eps dans la direction u = (cos phi, sin phi) et on regarde
    de combien la loss augmente :
        ordonnée = Delta L(phi) = L(theta* + eps*u) - L(theta*)
                 = 1/2 * eps^2 * u^T H u
                 = 1/2 * eps^2 * (2a*cos^2(phi) + 2b*sin^2(phi))
    En abscisse : l'angle phi de la direction, en degrés (0° = axe theta1, 90° = axe theta2).

    inputs :
        a   (float) : coefficient devant theta1^2
        b   (float) : coefficient devant theta2^2
        eps (float) : longueur du pas fait depuis le minimum
        n   (int)   : nombre d'angles phi entre 0° et 360°
    output :
        fig (go.Figure) : figure Plotly de Delta L en fonction de phi
    """
    phi_deg = np.linspace(0, 360, n)
    phi = phi_deg * np.pi / 180

    # hausse de la loss pour un pas eps dans chaque direction
    delta_L = np.zeros(n)
    for i in range(n):
        theta1 = eps * np.cos(phi[i])
        theta2 = eps * np.sin(phi[i])
        delta_L[i] = loss(theta1, theta2, a, b) - loss(0.0, 0.0, a, b)

    # valeurs extrêmes : direction plate (phi = 0°) et direction raide (phi = 90°)
    dL_plate = 0.5 * eps**2 * (2 * a)
    dL_raide = 0.5 * eps**2 * (2 * b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phi_deg, y=delta_L, mode="lines", line=dict(color="darkorange", width=3),
        name="ΔL(φ)",
        hovertemplate="φ = %{x:.0f}°<br>ΔL = %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 180, 360], y=[dL_plate, dL_plate, dL_plate], mode="markers",
        marker=dict(size=10, color="royalblue"),
        name=f"direction plate θ₁ : ΔL = {dL_plate:g}",
    ))
    fig.add_trace(go.Scatter(
        x=[90, 270], y=[dL_raide, dL_raide], mode="markers",
        marker=dict(size=10, color="crimson"),
        name=f"direction raide θ₂ : ΔL = {dL_raide:g}",
    ))

    fig.update_xaxes(title_text="φ (degrés) : direction u = (cos φ, sin φ)",
                     tickvals=[0, 45, 90, 135, 180, 225, 270, 315, 360])
    fig.update_yaxes(title_text="ΔL(φ) = L(θ* + ε·u) − L(θ*) = ½·ε²·uᵀHu")
    fig.update_layout(
        title=(f"Sensibilité de L = {a:g}·θ₁² + {b:g}·θ₂² : hausse de la loss "
               f"pour un pas ε = {eps:g} depuis le minimum"),
        width=1000, height=500,
    )
    return fig


def flatness_box(a=0.1, b=10.0, tau=1.8):
    """Calcule la boîte de platitude de Hochreiter & Schmidhuber (1997)
    pour L = a*theta1^2 + b*theta2^2.
    Chaque direction contribue autant à la hausse de la loss au coin de la boîte :
    1/2 * H_ii * dtheta_i^2 = tau / d, d'où dtheta_i = sqrt(2 tau / (d H_ii)).

    inputs :
        a   (float) : coefficient devant theta1^2
        b   (float) : coefficient devant theta2^2
        tau (float) : tolérance sur la loss (L doit rester <= tau dans la boîte)
    output :
        dtheta (np.ndarray, shape (2,)) : demi-côtés de la boîte (dtheta_1, dtheta_2)
    """
    H, eigvals, eigvecs = hessian(a, b)
    d = H.shape[0]                       # nombre de paramètres (ici 2)

    dtheta = np.zeros(d)
    for i in range(d):
        dtheta[i] = np.sqrt(2 * tau / (d * H[i, i]))
    return dtheta


def flatness_measures(a=0.1, b=10.0, tau=1.8):
    """Calcule trois mesures de platitude à partir de la Hessienne H.
    Plus ces valeurs sont petites, plus le minimum est plat.

    inputs :
        a   (float) : coefficient devant theta1^2
        b   (float) : coefficient devant theta2^2
        tau (float) : tolérance utilisée pour la boîte de platitude
    output :
        mesures (dict) : dictionnaire avec les clés
            "lambda_max" (float)      : plus grande valeur propre de H (direction la plus raide)
            "trace"      (float)      : tr(H) = somme des valeurs propres (courbure totale)
            "B_tilde"    (float)      : score de Hochreiter & Schmidhuber, -sum_i log(dtheta_i)
            "dtheta"     (np.ndarray) : demi-côtés de la boîte
            "volume"     (float)      : volume de la boîte = produit des côtés 2*dtheta_i
    """
    H, eigvals, eigvecs = hessian(a, b)
    dtheta = flatness_box(a, b, tau)

    lambda_max = eigvals[-1]             # les valeurs propres sont rangées par ordre croissant
    trace = H[0, 0] + H[1, 1]

    B_tilde = 0.0
    volume = 1.0
    for i in range(len(dtheta)):
        B_tilde = B_tilde - np.log(dtheta[i])
        volume = volume * (2 * dtheta[i])   # un côté de la boîte mesure 2*dtheta_i

    mesures = {
        "lambda_max": lambda_max,
        "trace": trace,
        "B_tilde": B_tilde,
        "dtheta": dtheta,
        "volume": volume,
    }
    return mesures


def plot_flatness_box(a=0.1, b=10.0, tau=1.8, margin=1.35, n=150):
    """Trace en 3D la loss L = a*theta1^2 + b*theta2^2 et la boîte de platitude
    de Hochreiter & Schmidhuber (1997) pour la tolérance tau :
    - la surface de L (bleu) et la partie de la surface au-dessus de la boîte (orange) ;
    - la boîte au sol (pointillés noirs) et L le long de son bord (ligne orange) ;
    - le plan de tolérance L = tau (rouge) et les coins, où L atteint exactement tau.
    Le titre donne Delta theta_i, le volume et le score B~.

    inputs :
        a      (float) : coefficient devant theta1^2
        b      (float) : coefficient devant theta2^2
        tau    (float) : tolérance sur la loss
        margin (float) : la surface est tracée sur margin fois la taille de la boîte
        n      (int)   : nombre de points par axe pour la surface
    output :
        fig (go.Figure) : figure Plotly 3D
    """
    m = flatness_measures(a, b, tau)
    dt1 = m["dtheta"][0]
    dt2 = m["dtheta"][1]

    # ----- surface de la loss -----
    l1 = margin * dt1
    l2 = margin * dt2
    t1 = np.linspace(-l1, l1, n)
    t2 = np.linspace(-l2, l2, n)
    T1, T2 = np.meshgrid(t1, t2)
    Z = loss(T1, T2, a, b)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=t1, y=t2, z=Z, colorscale="Blues", reversescale=True, opacity=0.5,
        showscale=False, name="loss L(θ)", showlegend=True,
        hovertemplate="θ₁ = %{x:.2f}<br>θ₂ = %{y:.3f}<br>L = %{z:.3f}<extra>L(θ)</extra>",
    ))

    # ----- partie de la surface au-dessus de la boîte -----
    b1 = np.linspace(-dt1, dt1, 60)
    b2 = np.linspace(-dt2, dt2, 60)
    B1, B2 = np.meshgrid(b1, b2)
    ZB = loss(B1, B2, a, b) + 1e-3      # petit décalage pour qu'elle passe au-dessus
    fig.add_trace(go.Surface(
        x=b1, y=b2, z=ZB,
        colorscale=[[0, "darkorange"], [1, "darkorange"]], opacity=0.9,
        showscale=False, name="loss au-dessus de la boîte (L ≤ τ)", showlegend=True,
        hovertemplate="θ₁ = %{x:.2f}<br>θ₂ = %{y:.3f}<br>L = %{z:.3f}<extra>dans la boîte</extra>",
    ))

    # ----- plan de tolérance L = tau -----
    fig.add_trace(go.Surface(
        x=[-l1, l1], y=[-l2, l2], z=[[tau, tau], [tau, tau]],
        colorscale=[[0, "crimson"], [1, "crimson"]], opacity=0.15,
        showscale=False, name=f"plan de tolérance L = τ = {tau:g}", showlegend=True,
        hoverinfo="skip",
    ))

    # ----- bord de la boîte : on parcourt les 4 côtés l'un après l'autre -----
    coins = [(-dt1, -dt2), (dt1, -dt2), (dt1, dt2), (-dt1, dt2), (-dt1, -dt2)]
    e1 = []
    e2 = []
    for j in range(4):
        x_debut, y_debut = coins[j]
        x_fin, y_fin = coins[j + 1]
        for s in np.linspace(0, 1, 50):
            e1.append(x_debut + s * (x_fin - x_debut))
            e2.append(y_debut + s * (y_fin - y_debut))
    e1 = np.array(e1)
    e2 = np.array(e2)

    fig.add_trace(go.Scatter3d(
        x=e1, y=e2, z=np.zeros(len(e1)), mode="lines",
        line=dict(color="black", width=5, dash="dash"),
        name=f"boîte [−{dt1:.2f}, {dt1:.2f}] × [−{dt2:.2f}, {dt2:.2f}]",
    ))
    fig.add_trace(go.Scatter3d(
        x=e1, y=e2, z=loss(e1, e2, a, b), mode="lines",
        line=dict(color="darkorange", width=7), name="loss sur le bord de la boîte",
    ))

    # ----- segments verticaux aux 4 coins (L = tau) -----
    for k in range(4):
        c1, c2 = coins[k]
        fig.add_trace(go.Scatter3d(
            x=[c1, c1], y=[c2, c2], z=[0, tau], mode="lines+markers",
            line=dict(color="crimson", width=4), marker=dict(size=[0, 5], color="crimson"),
            name="coins de la boîte : L = τ", showlegend=(k == 0), legendgroup="coins",
            hovertemplate=f"coin ({c1:+.2f}, {c2:+.3f})<br>L = τ = {tau:g}<extra></extra>",
        ))

    # ----- demi-côtés et minimum -----
    fig.add_trace(go.Scatter3d(
        x=[0, dt1], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="black", width=6),
        name=f"Δθ₁ = {dt1:.2f}",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, dt2], z=[0, 0], mode="lines", line=dict(color="dimgray", width=6),
        name=f"Δθ₂ = {dt2:.2f}",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=6, color="black"),
        name="minimum θ*",
    ))

    fig.update_layout(
        title=dict(
            text=(f"Boîte de platitude pour L = {a:g}·θ₁² + {b:g}·θ₂²,  tolérance τ = {tau:g}"
                  f"<br><sup>Δθ₁ = {dt1:.2f},  Δθ₂ = {dt2:.2f},  volume V = {m['volume']:.2f}"
                  f"  →  score B̃ = −Σ log Δθᵢ = {m['B_tilde']:.3f}"
                  f"   |   λ_max = {m['lambda_max']:g},  tr(H) = {m['trace']:g}</sup>"),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="θ₁ (direction plate)", yaxis_title="θ₂ (direction raide)",
            zaxis_title="L(θ)", zaxis=dict(range=[0, Z.max()]),
            aspectmode="manual", aspectratio=dict(x=1.6, y=1, z=0.8),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
            annotations=[
                dict(x=dt1, y=dt2, z=tau, text="L = τ", showarrow=True, arrowcolor="crimson",
                     font=dict(color="crimson"), ax=30, ay=-30),
                dict(x=dt1 / 2, y=0, z=0, text=f"Δθ₁ = {dt1:.2f}", showarrow=False, yshift=12),
                dict(x=0, y=dt2 / 2, z=0, text=f"Δθ₂ = {dt2:.2f}", showarrow=False, xshift=40),
            ],
        ),
        legend=dict(x=0.0, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        width=1000, height=700, margin=dict(t=100),
    )
    return fig
