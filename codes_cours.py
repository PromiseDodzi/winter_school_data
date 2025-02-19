import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

def clean_text(text):
    # Supprimer les emojis
    text = re.sub(r'[^\w\s,]', '', text)  # Garde uniquement les caractères alphanumériques et les espaces

    # Supprimer les mentions d'utilisateur (ex: @AmadeusOfficiel)
    text = re.sub(r'@\w+', '', text)

    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Supprimer les hashtags (ex: #Jëli)
    text = re.sub(r'#\w+', '', text)

    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()

    # Convertir le texte en minuscules
    text = text.lower()

    return text

if __name__ == "__main__":
    # Texte à nettoyer
    mon_texte = "🔥 Trop cool le show d’hier avec @AmadeusOfficiel et @officielwallyseck😍 ! J’ai trop hâte de te voir 💯 à ziguinchor!!! #Jëli #Mbëggeeldafaneex # https://t.co/abc123efg "
    cleaned_text = clean_text(mon_texte)
    print(cleaned_text)

    # N-grams
    text_gram = "Je suis là"
    def n_grams(text, n):
        n_grams = list(zip(text.split(), text.split()[n-1:]))
        return n_grams
    print(n_grams(text_gram, 2))

    # Illustration de la loi de Zipf
    texte = """
    Le soleil brillait fort sur la mer bleue. Les vagues venaient lécher le sable doré,
    et les mouettes planaient dans le ciel. Le vent soufflait doucement, apportant avec lui
    la odeur salée de le océan. Les enfants jouaient en riant, construisant des châteaux de sable
    tandis que les adultes profitaient de le soleil.
    """

    # Nettoyage et tokenisation du texte
    mots = re.findall(r'\b\w+\b', texte.lower())

    # Compter la fréquence des mots
    frequence_mots = Counter(mots)

    # Trier les mots par fréquence décroissante
    mots_tries = frequence_mots.most_common()

    # Récupérer les fréquences et les mots pour l'affichage
    frequences = [freq for mot, freq in mots_tries]
    rangs = np.arange(1, len(frequences) + 1)
    mots_texte = [mot for mot, freq in mots_tries]

    # Tracer le graphique en échelle logarithmique
    plt.figure(figsize=(12, 8))
    plt.loglog(rangs, frequences, marker="o", linestyle="none", color='b')
    plt.xlabel("Rang du mot")
    plt.ylabel("Fréquence du mot")
    plt.title("Illustration de la loi de puissance")
    plt.grid(True, which="both", linestyle="--", lw=0.5)

    # Ajouter les mots les plus fréquents au graphique, en vertical
    nb_mots_affiches = 20  # Nombre de mots à afficher
    for i in range(nb_mots_affiches):
        plt.text(rangs[i], frequences[i], mots_texte[i], fontsize=10, ha='right', rotation=90)

    plt.savefig("loi_puissance.png")
    plt.show()

    # Illustration de la loi de Heaps
    corpus = texte * 100  # Répéter pour simuler un corpus plus large

    # Tokeniser le corpus en mots
    words = re.findall(r'\b\w+\b', corpus.lower())

    tokens = []
    types_set = set()
    types_counts = []
    tokens_counts = []

    # Simulation de la lecture du corpus mot par mot
    for word in words:
        tokens.append(word)
        types_set.add(word)
        tokens_counts.append(len(tokens))
        types_counts.append(len(types_set))

    # Tracer le graphique illustrant la loi de Heaps
    plt.figure(figsize=(12, 8))
    plt.plot(tokens_counts, types_counts, label="Évolution du vocabulaire", color="b")
    plt.xlabel("Nombre de tokens")
    plt.ylabel("Nombre de types")
    plt.title("Illustration de la loi de Heaps")
    plt.grid(True, which="both", linestyle="--", lw=0.5)

    k = 10
    beta = 0.5
    heaps_curve = [k * (n ** beta) for n in tokens_counts]
    plt.plot(tokens_counts, heaps_curve, label="Loi de Heaps (théorique)", color="red", linestyle="--")

    plt.legend()
    plt.savefig("loi_heaps.png")
    plt.show()

    # Illustration de la loi de Brièveté
    words_2 = texte.lower().split()
    word_counts = Counter(words_2)

    word_lengths = []
    frequencies = []
    word_labels = []

    for word_1, count in word_counts.items():
        word_lengths.append(len(word_1))
        frequencies.append(count)
        word_labels.append(word_1)

    plt.figure(figsize=(10, 6))
    plt.scatter(word_lengths, frequencies)

    for i_, word_ in enumerate(word_labels):
        plt.text(word_lengths[i_] + 0.1, frequencies[i_], word_, fontsize=9, rotation=90)

    plt.xlabel('Longueur de mot')
    plt.ylabel('Fréquence de mot')
    plt.title('Longueur de mot vs Fréquence')
    plt.grid(True)
    plt.savefig("loi_brieveté.png")
    plt.show()
