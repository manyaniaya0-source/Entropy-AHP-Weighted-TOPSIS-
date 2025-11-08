import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Entropy-AHP TOPSIS", layout="wide", page_icon="📊")

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal avec style
st.markdown('<p class="main-header">🎯 Entropy-AHP Weighted TOPSIS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Système d\'aide à la décision multicritère avancé</p>', unsafe_allow_html=True)

# Sidebar amélioré
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/decision.png", width=100)
    st.title("📋 Navigation")
    
    page = st.radio("", [
        "🏠 Accueil",
        "1️⃣ Matrice de décision",
        "2️⃣ Poids Entropy",
        "3️⃣ Poids AHP",
        "4️⃣ Poids combinés",
        "5️⃣ Calcul TOPSIS",
        "📊 Résultats & Analyse",
        "ℹ️ Guide d'utilisation"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Indicateur de progression
    if 'decision_matrix' in st.session_state and st.session_state.decision_matrix is not None:
        progress = 0
        if st.session_state.decision_matrix is not None: progress += 20
        if 'entropy_weights' in st.session_state and st.session_state.entropy_weights is not None: progress += 20
        if 'ahp_weights' in st.session_state and st.session_state.ahp_weights is not None: progress += 20
        if 'combined_weights' in st.session_state and st.session_state.combined_weights is not None: progress += 20
        if 'topsis_results' in st.session_state and st.session_state.topsis_results is not None: progress += 20
        
        st.markdown("### 📈 Progression")
        st.progress(progress / 100)
        st.write(f"{progress}% complété")

# Initialisation des variables de session
for key in ['decision_matrix', 'criteria_names', 'alternative_names', 'criteria_types', 
            'entropy_weights', 'ahp_weights', 'combined_weights', 'topsis_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# ==================== FONCTIONS DE CALCUL ====================

def normalize_matrix(matrix):
    """Normalisation vectorielle de la matrice"""
    return matrix / np.sqrt((matrix ** 2).sum(axis=0))

def calculate_entropy_weights(matrix):
    """Calcul des poids par la méthode Entropy"""
    m, n = matrix.shape
    p = matrix / matrix.sum(axis=0)
    epsilon = 1e-10
    p_safe = np.where(p > 0, p, epsilon)
    e = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(m)
    b = 1 - e
    w = b / b.sum()
    return w, e, b

def calculate_ahp_weights_from_matrix(comparison_matrix):
    """Calcul des poids AHP avec vérification de cohérence"""
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_idx = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_idx].real
    weights = np.abs(weights) / np.abs(weights).sum()
    
    lambda_max = eigenvalues[max_idx].real
    n = len(comparison_matrix)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    
    return weights, CR, lambda_max

def combine_weights(entropy_weights, ahp_weights):
    """Combinaison multiplicative des poids"""
    combined = (entropy_weights * ahp_weights)
    return combined / combined.sum()

def calculate_topsis(matrix, weights, criteria_types):
    """Algorithme TOPSIS complet"""
    normalized = normalize_matrix(matrix)
    weighted = normalized * weights
    
    ideal_positive = np.zeros(len(weights))
    ideal_negative = np.zeros(len(weights))
    
    for j, ctype in enumerate(criteria_types):
        if ctype == 'benefit':
            ideal_positive[j] = weighted[:, j].max()
            ideal_negative[j] = weighted[:, j].min()
        else:
            ideal_positive[j] = weighted[:, j].min()
            ideal_negative[j] = weighted[:, j].max()
    
    distance_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))
    closeness = distance_negative / (distance_positive + distance_negative)
    
    return closeness, distance_positive, distance_negative, weighted, ideal_positive, ideal_negative

# ==================== PAGE ACCUEIL ====================

if page == "🏠 Accueil":
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://img.icons8.com/clouds/400/000000/analytics.png", width=200)
    
    st.markdown("---")
    
    # Cartes de fonctionnalités
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
            <h2>🎯 ENTROPY</h2>
            <p>Poids objectifs basés sur la variabilité des données</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
            <h2>⚖️ AHP</h2>
            <p>Poids subjectifs basés sur l'expertise des décideurs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
            <h2>📊 TOPSIS</h2>
            <p>Classement optimal par similarité aux solutions idéales</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Avantages
    st.markdown("### 🌟 Pourquoi utiliser cette méthode ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **Réduction des biais**: Combine objectivité et expertise")
        st.success("✅ **Méthode validée**: Basée sur des recherches scientifiques")
        st.success("✅ **Flexibilité**: Applicable à divers domaines de décision")
    
    with col2:
        st.info("📌 **Applications**: Sélection de fournisseurs, projets, localisation...")
        st.info("📌 **Interface intuitive**: Processus guidé étape par étape")
        st.info("📌 **Analyse complète**: Visualisations et analyse de sensibilité")
    
    st.markdown("---")
    
    # Exemple rapide
    with st.expander("📚 Voir un exemple d'utilisation"):
        st.markdown("""
        **Exemple: Sélection de fournisseurs de matériaux de construction**
        
        1. **Matrice de décision**: 5 fournisseurs (A1-A5) évalués sur 7 critères
           - Taux de produits qualifiés, Prix, Part de marché, etc.
        
        2. **Calcul Entropy**: Identifie les critères avec plus de variabilité
        
        3. **Calcul AHP**: Les experts évaluent l'importance relative des critères
        
        4. **Combinaison**: Fusion des deux approches pour des poids équilibrés
        
        5. **TOPSIS**: Classement final basé sur la proximité aux solutions idéales
        
        **Résultat**: Le fournisseur A1 obtient le meilleur score (φ = 0.6395)
        """)
    
    st.markdown("---")
    
    # Bouton de démarrage
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 COMMENCER L'ANALYSE", type="primary", use_container_width=True):
            st.session_state.page = "1️⃣ Matrice de décision"
            st.rerun()

# ==================== PAGE 1: MATRICE DE DÉCISION ====================

elif page == "1️⃣ Matrice de décision":
    st.header("📝 Étape 1: Construction de la matrice de décision")
    
    tab1, tab2, tab3 = st.tabs(["✍️ Saisie manuelle", "📤 Import fichier", "📋 Exemple pré-rempli"])
    
    # TAB 1: Saisie manuelle
    with tab1:
        st.markdown("### Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_alternatives = st.number_input("🔢 Nombre d'alternatives", 
                                            min_value=2, max_value=20, value=5)
        with col2:
            n_criteria = st.number_input("🔢 Nombre de critères", 
                                        min_value=2, max_value=15, value=7)
        
        st.markdown("### 📌 Noms des alternatives")
        alternative_names = []
        cols = st.columns(min(5, n_alternatives))
        for i in range(n_alternatives):
            with cols[i % 5]:
                name = st.text_input(f"Alt. {i+1}", value=f"A{i+1}", 
                                    key=f"alt_{i}", label_visibility="collapsed")
                alternative_names.append(name)
        
        st.markdown("### 📊 Configuration des critères")
        criteria_names = []
        criteria_types = []
        
        for j in range(n_criteria):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f"**C{j+1}**")
            with col2:
                name = st.text_input(f"Nom", value=f"Critère {j+1}", 
                                    key=f"crit_{j}", label_visibility="collapsed")
                criteria_names.append(name)
            with col3:
                ctype = st.selectbox("Type", ["benefit", "cost"], 
                                    key=f"type_{j}", label_visibility="collapsed")
                criteria_types.append(ctype)
        
        st.markdown("### 📋 Valeurs de la matrice")
        st.info("💡 Saisissez les valeurs pour chaque alternative et critère")
        
        matrix_data = []
        for i in range(n_alternatives):
            st.markdown(f"**{alternative_names[i]}**")
            row = []
            cols = st.columns(n_criteria)
            for j in range(n_criteria):
                with cols[j]:
                    val = st.number_input(criteria_names[j], 
                                         value=float(np.random.randint(10, 100)),
                                         key=f"val_{i}_{j}",
                                         label_visibility="collapsed",
                                         format="%.2f")
                    row.append(val)
            matrix_data.append(row)
        
        if st.button("✅ Valider et continuer", type="primary", use_container_width=True):
            st.session_state.decision_matrix = np.array(matrix_data, dtype=float)
            st.session_state.criteria_names = criteria_names
            st.session_state.alternative_names = alternative_names
            st.session_state.criteria_types = criteria_types
            st.success("✅ Matrice enregistrée avec succès!")
            st.balloons()
    
    # TAB 2: Import fichier
    with tab2:
        st.markdown("### 📤 Importer depuis un fichier")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Format attendu:")
            st.markdown("""
            - **Ligne 1**: Noms des critères
            - **Colonne 1**: Noms des alternatives
            - **Cellules**: Valeurs numériques
            
            Formats acceptés: `.xlsx`, `.csv`
            """)
        
        with col2:
            # Créer un exemple téléchargeable
            example_df = pd.DataFrame({
                'Critère1': [95, 98, 93, 91, 92],
                'Critère2': [36, 39, 33, 37, 35],
                'Critère3': [19, 17, 21, 23, 16]
            }, index=['Alt1', 'Alt2', 'Alt3', 'Alt4', 'Alt5'])
            
            excel_buffer = BytesIO()
            example_df.to_excel(excel_buffer, engine='openpyxl')
            
            st.download_button(
                label="📥 Télécharger un modèle Excel",
                data=excel_buffer.getvalue(),
                file_name="modele_matrice.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        uploaded_file = st.file_uploader("Choisir un fichier", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, index_col=0)
                else:
                    df = pd.read_excel(uploaded_file, index_col=0)
                
                st.success("✅ Fichier chargé avec succès!")
                st.dataframe(df, use_container_width=True)
                
                st.markdown("### Configuration des types de critères")
                criteria_types = []
                cols = st.columns(len(df.columns))
                for idx, col in enumerate(df.columns):
                    with cols[idx]:
                        ctype = st.selectbox(col, ["benefit", "cost"], 
                                           key=f"upload_type_{idx}")
                        criteria_types.append(ctype)
                
                if st.button("✅ Valider l'import", type="primary", use_container_width=True):
                    st.session_state.decision_matrix = df.values
                    st.session_state.criteria_names = list(df.columns)
                    st.session_state.alternative_names = list(df.index)
                    st.session_state.criteria_types = criteria_types
                    st.success("✅ Données importées!")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Erreur lors de l'import: {e}")
    
    # TAB 3: Exemple pré-rempli
    with tab3:
        st.markdown("### 📋 Charger l'exemple du papier de recherche")
        st.info("Exemple de sélection de fournisseurs de matériaux de construction")
        
        example_data = {
            'Taux produits qualifiés (%)': [0.95, 0.98, 0.93, 0.91, 0.92],
            'Prix (k$)': [36, 39, 33, 37, 35],
            'Part de marché (%)': [0.19, 0.17, 0.21, 0.23, 0.16],
            'Capacité (kg/temps)': [53, 52, 57, 56, 51],
            'Taux nouveaux produits (%)': [0.73, 0.75, 0.69, 0.77, 0.76],
            'Délai livraison (jours)': [11, 13, 11, 12, 10],
            'Taux livraison à temps (%)': [0.93, 0.89, 0.92, 0.87, 0.86]
        }
        
        example_df = pd.DataFrame(example_data, 
                                 index=['Fournisseur A', 'Fournisseur B', 
                                       'Fournisseur C', 'Fournisseur D', 
                                       'Fournisseur E'])
        
        st.dataframe(example_df, use_container_width=True)
        
        example_types = ['benefit', 'cost', 'benefit', 'benefit', 'benefit', 'cost', 'benefit']
        
        if st.button("📥 Charger cet exemple", type="primary", use_container_width=True):
            st.session_state.decision_matrix = example_df.values
            st.session_state.criteria_names = list(example_df.columns)
            st.session_state.alternative_names = list(example_df.index)
            st.session_state.criteria_types = example_types
            st.success("✅ Exemple chargé!")
            st.balloons()
    
    # Affichage de la matrice enregistrée
    if st.session_state.decision_matrix is not None:
        st.markdown("---")
        st.markdown("### 📊 Matrice enregistrée")
        
        df_display = pd.DataFrame(
            st.session_state.decision_matrix,
            columns=st.session_state.criteria_names,
            index=st.session_state.alternative_names
        )
        
        # Ajouter une ligne avec les types de critères
        types_row = pd.DataFrame([st.session_state.criteria_types], 
                                columns=st.session_state.criteria_names,
                                index=['Type'])
        
        st.dataframe(types_row.style.background_gradient(axis=1, cmap='coolwarm'), 
                    use_container_width=True)
        st.dataframe(df_display.style.background_gradient(axis=0, cmap='YlGnBu'), 
                    use_container_width=True)

# ==================== PAGE 2: POIDS ENTROPY ====================

elif page == "2️⃣ Poids Entropy":
    st.header("🔢 Étape 2: Calcul des poids Entropy (objectifs)")
    
    if st.session_state.decision_matrix is None:
        st.warning("⚠️ Veuillez d'abord configurer la matrice de décision (Étape 1)")
        if st.button("← Retour à l'étape 1"):
            st.rerun()
    else:
        st.markdown("""
        ### 📖 Principe de la méthode Entropy
        
        La méthode **Entropy** calcule des poids **objectifs** basés sur la **variabilité** des données:
        
        - 📊 **Plus un critère varie** entre les alternatives, plus il est informatif → poids élevé
        - 📏 **Moins un critère varie**, moins il discrimine → poids faible
        - 🎯 **Formule**: $w_j = \\frac{1 - e_j}{n - \\sum e_j}$ où $e_j$ est l'entropie du critère j
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔄 Calculer les poids Entropy", type="primary", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    weights, entropy_values, utility_values = calculate_entropy_weights(
                        st.session_state.decision_matrix
                    )
                    st.session_state.entropy_weights = weights
                    st.success("✅ Calcul terminé!")
        
        if st.session_state.entropy_weights is not None:
            weights = st.session_state.entropy_weights
            _, entropy_values, utility_values = calculate_entropy_weights(
                st.session_state.decision_matrix
            )
            
            # Tableau des résultats
            st.markdown("### 📊 Résultats détaillés")
            
            results_df = pd.DataFrame({
                'Critère': st.session_state.criteria_names,
                'Entropie (e_j)': entropy_values,
                'Utilité (1-e_j)': utility_values,
                'Poids Entropy': weights,
                'Poids (%)': weights * 100
            })
            
            st.dataframe(results_df.style.format({
                'Entropie (e_j)': '{:.4f}',
                'Utilité (1-e_j)': '{:.4f}',
                'Poids Entropy': '{:.4f}',
                'Poids (%)': '{:.2f}%'
            }).background_gradient(subset=['Poids Entropy'], cmap='Greens'),
            use_container_width=True)
            
            # Visualisations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(results_df, x='Critère', y='Poids Entropy',
                            title="📊 Distribution des poids Entropy",
                            color='Poids Entropy',
                            color_continuous_scale='Blues',
                            text='Poids (%)')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(results_df, values='Poids Entropy', names='Critère',
                            title="🎯 Répartition des poids Entropy",
                            hole=0.4)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation
            st.markdown("### 💡 Interprétation")
            
            max_idx = np.argmax(weights)
            min_idx = np.argmin(weights)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Critère le plus important", 
                         st.session_state.criteria_names[max_idx],
                         f"{weights[max_idx]*100:.2f}%")
            
            with col2:
                st.metric("📉 Critère le moins important",
                         st.session_state.criteria_names[min_idx],
                         f"{weights[min_idx]*100:.2f}%")
            
            with col3:
                st.metric("📊 Écart-type", 
                         f"{np.std(weights):.4f}",
                         "Dispersion des poids")
            
            st.info(f"""
            ℹ️ **Analyse**: Le critère **{st.session_state.criteria_names[max_idx]}** 
            présente la plus grande variabilité entre les alternatives, ce qui le rend 
            le plus discriminant objectivement. À l'inverse, **{st.session_state.criteria_names[min_idx]}** 
            varie peu et apporte moins d'information pour la décision.
            """)

# ==================== PAGE 3: POIDS AHP ====================

elif page == "3️⃣ Poids AHP":
    st.header("⚖️ Étape 3: Calcul des poids AHP (subjectifs)")
    
    if st.session_state.decision_matrix is None:
        st.warning("⚠️ Veuillez d'abord configurer la matrice de décision (Étape 1)")
    else:
        st.markdown("""
        ### 📖 Principe de la méthode AHP
        
        La méthode **AHP** (Analytic Hierarchy Process) utilise des **comparaisons par paires** 
        pour déterminer les poids **subjectifs** basés sur l'expertise:
        
        - 🤝 Comparer l'importance relative de chaque paire de critères
        - 🎚️ Utiliser l'échelle de Saaty (1-9)
        - ✅ Vérifier la cohérence des jugements (CR < 0.10)
        """)
        
        n_criteria = len(st.session_state.criteria_names)
        
        tab1, tab2 = st.tabs(["🔢 Matrice de comparaison", "✏️ Saisie directe"])
        
        # TAB 1: Matrice de comparaison par paires
        with tab1:
            st.markdown("### 📊 Échelle de Saaty")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                scale_df = pd.DataFrame({
                    'Valeur': [1, 3, 5, 7, 9, 2,4,6,8],
                    'Signification': [
                        'Égale importance',
                        'Importance modérée',
                        'Importance forte',
                        'Importance très forte',
                        'Importance extrême',
                        'Valeur intermédiaire',
                        'Valeur intermédiaire',
                        'Valeur intermédiaire',
                        'Valeur intermédiaire'
                    ]
                })
                st.dataframe(scale_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.info("""
                **Comment remplir?**
                
                Pour chaque paire (Critère A, Critère B):
                - Si A est **plus important** que B: valeur > 1
                - Si A est **moins important** que B: valeur < 1
                - Si A et B sont **égaux**: valeur = 1
                
                La matrice est automatiquement symétrique!
                """)
            
            st.markdown("### 🔢 Saisissez les comparaisons")
            
            comparison_matrix = np.ones((n_criteria, n_criteria))
            
            # Affichage sous forme de tableau interactif
            comparisons = []
            for i in range(n_criteria):
                for j in range(i+1, n_criteria):
                    comparisons.append((i, j))
            
            n_cols = 3
            for idx in range(0, len(comparisons), n_cols):
                cols = st.columns(n_cols)
                for col_idx, col in enumerate(cols):
                    if idx + col_idx < len(comparisons):
                        i, j = comparisons[idx + col_idx]
                        with col:
                            val = st.select_slider(
                                f"{st.session_state.criteria_names[i]} vs {st.session_state.criteria_names[j]}",
                                options=[1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9],
                                value=1,
                                key=f"ahp_{i}_{j}",
                                format_func=lambda x: f"{x:.2f}" if x < 1 else f"{int(x)}"
                            )
                            comparison_matrix[i, j] = val
                            comparison_matrix[j, i] = 1/val
            
            st.markdown("### 📊 Matrice de comparaison complète")
            comparison_df = pd.DataFrame(
                comparison_matrix,
                columns=st.session_state.criteria_names,
                index=st.session_state.criteria_names
            )
            st.dataframe(comparison_df.style.format("{:.2f}").background_gradient(cmap='RdYlGn', axis=None),
                        use_container_width=True)