import os
import PyPDF2

def extract_text_from_pdfs(pdf_path): 
    """
    Extrait le texte de chaque page des fichiers PDF fournis,
    et combine tout le texte en une seule chaîne.
    """
    all_text = ""

    for pdf_path in pdf_path:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_text = ""

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"

            print(f" {os.path.basename(pdf_path)} : {len(reader.pages)} pages extraites.")
            all_text += pdf_text + "\n"

    return all_text

# === Exemple d'utilisation ===

# Liste de fichiers PDF à traiter ( un document officiel)
pdf_files = ["ich-guideline-good-clinical-practice-e6r2.pdf"]

# Extraction du texte
texte_total = extract_text_from_pdfs(pdf_files)

# Aperçu dans le terminal
print("\n Aperçu du texte extrait :\n")
print(texte_total[:1000])  # Affiche les 1000 premiers caractères seulement