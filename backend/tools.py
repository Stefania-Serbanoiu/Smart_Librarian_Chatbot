import base64
from pathlib import Path
from typing import Optional
import pyttsx3
from openai import OpenAI
from .config import OPENAI_API_KEY, IMAGE_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


# Simple local dictionary for full summaries (tool data source)
_FULL_SUMMARIES = {
    "The Hobbit": (
        "Bilbo Baggins, un hobbit comod, este recrutat de pitici și de Gandalf pentru a recupera "
        "comoara furată de dragonul Smaug. Călătoria îl poartă prin păduri întunecate, peșteri și "
        "întâlniri cu creaturi primejdioase, dar îl ajută să-și descopere curajul și istețimea."
    ),
    "1984": (
        "O societate distopică în care Big Brother supraveghează tot. Winston Smith încearcă să-și "
        "păstreze libertatea interioară într-un sistem care manipulează adevărul și limbajul."
    ),
    "To Kill a Mockingbird": (
        "Prin ochii lui Scout, aflăm despre un proces nedrept și rasismul sistemic. Atticus Finch "
        "întruchipează integritatea, învățându-și copiii compasiunea și curajul moral."
    ),
    "Brave New World": (
        "O lume „perfectă” menținută prin condiționare, droguri și divertisment. Fericirea standardizată "
        "se lovește de dorința individului pentru sens și autenticitate."
    ),
    "Harry Potter and the Philosopher's Stone": (
        "Harry descoperă că este vrăjitor, intră la Hogwarts și își face primii prieteni apropiați. "
        "Împreună descoperă misterul Pietrei Filozofale și înfruntă primele umbre ale lui Voldemort."
    ),
    "The Lord of the Rings: The Fellowship of the Ring": (
        "Frăția Inelului pornește din Comitat pentru a-l proteja pe Frodo. Încercările consolidează "
        "loialități, dar și pun la încercare voința fiecăruia."
    ),
    "The Book Thief": (
        "Liesel fură cărți ca refugiu într-o Germanie aflată în război. Învățarea și poveștile "
        "devin o formă de rezistență intimă."
    ),
    "The Catcher in the Rye": (
        "Holden călătorește prin New York într-o căutare dezordonată a autenticității, departe de ceea "
        "ce consideră ipocrizie."
    ),
    "Fahrenheit 451": (
        "Guy Montag, pompier care arde cărți, trece printr-o criză de conștiință și caută sens într-o "
        "societate care evită gândirea critică."
    ),
    "The Kite Runner": (
        "Amir își înfruntă vina din copilărie și caută iertarea, întorcându-se într-un Afganistan transformat "
        "de războaie."
    ),
    "Dune": (
        "Paul Atreides descoperă profeții și politici pe Arrakis. Lupta pentru spice se împletește cu destinul, "
        "ecologia și cultura fremenilor."
    ),
    "The Little Prince": (
        "Un băiețel de pe o planetă îndepărtată călătorește prin univers, întâlnind diverse personaje și "
        "învățând lecții despre viață, iubire și prietenie."
    ),
    "Crime and Punishment": (
        "Raskolnikov, un student sărac din Sankt Petersburg, comite o crimă pentru a-și demonstra teoria "
        "despre oameni extraordinari. Vinovăția și paranoia îl macină treptat, conducându-l spre o luptă "
        "morală și spirituală."
    ),
    "Pride and Prejudice": (
        "Elizabeth Bennet, inteligentă și independentă, navighează prejudecăți sociale și tensiuni de clasă. "
        "Întâlnirea cu mândrul Darcy scoate la iveală conflicte între orgoliu, dragoste și convenții sociale."
    ),
    "Moby-Dick": (
        "Căpitanul Ahab pornește într-o vânătoare obsesivă a balenei albe, simbol al destinului implacabil. "
        "Povestea explorează lupta omului cu natura, cu sine și cu ideea de fatalitate."
    ),
    "Anna Karenina": (
        "Anna, prinsă într-o căsnicie lipsită de iubire, trăiește o pasiune interzisă cu Vronski. Povestea "
        "ei tragică reflectă tensiunile dintre dorința personală și convențiile sociale ale Rusiei aristocratice."
    ),
    "The Great Gatsby": (
        "Jay Gatsby, milionar misterios, organizează petreceri somptuoase pentru a-și recâștiga iubirea pierdută. "
        "Povestea, narată de Nick Carraway, dezvăluie iluziile și decăderea visului american."
    ),
    "One Hundred Years of Solitude": (
        "Saga familiei Buendía în orașul Macondo dezvăluie cicluri de destin, iubire și pierdere. "
        "Realismul magic transformă istoria și experiențele personale într-o reflecție universală."
    ),
    "Frankenstein": (
        "Victor Frankenstein creează o creatură vie din părți moarte, dar refuză să-și accepte responsabilitatea. "
        "Monstrul, respins de societate, caută înțelegere și răzbunare, punând întrebări despre umanitate și știință."
    ),
    "Les Misérables": (
        "Jean Valjean, un fost condamnat, își caută mântuirea prin fapte bune, dar este urmărit neîncetat de Javert. "
        "Romanul explorează nedreptatea socială, compasiunea și sacrificiul."
    ),
    "The Alchemist": (
        "Santiago, un păstor andaluz, visează la o comoară ascunsă lângă piramide. Călătoria sa devine o metaforă "
        "pentru descoperirea propriei meniri și ascultarea inimii."
    ),
    "Dracula": (
        "Contele Dracula călătorește din Transilvania în Anglia pentru a-și răspândi puterea. "
        "Un grup de oameni curajoși luptă împotriva lui, într-o poveste despre frică, seducție și supranatural."
    ),
    "The Lord of the Rings": (
        "Frodo Baggins, un hobbit, pornește într-o călătorie epică pentru a distruge inelul lui Sauron. "
        "Povestea explorează teme de prietenie, sacrificiu și lupta între bine și rău."
    ),
    "The Name of the Wind": (
        "Kvothe, un tânăr talentat, își povestește viața plină de aventuri, magie și muzică, în căutarea adevărului despre trecutul său."
    ),
    "The Chronicles of Narnia": (
        "Cinci copii sunt transportați în lumea magică a Narniei, unde se alătură luptei împotriva regelui malefic."
    ),
    "All the light we cannot see": (
        "În timpul celui de-al Doilea Război Mondial, o fetiță oarbă din Franța și un băiat german se confruntă cu provocările și ororile războiului."
    )
}



def get_summary_by_title(title: str) -> str:
    # Case-insensitive exact match over keys
    for k, v in _FULL_SUMMARIES.items():
        if k.lower() == title.lower():
            return v
    # Fallback if not found
    return "Rezumat complet indisponibil pentru acest titlu în setul local."


def tts_save(text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",  
        input=text
    ) as response:
        response.stream_to_file(out_path)
    return out_path


def generate_book_image(title: str, themes: str, out_path: Path) -> Path:
    prompt = (
        f"Create a clean, suggestive book-cover style illustration for the book '{title}'. "
        f"Visual hints of themes: {themes}. Minimalist, modern composition."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1)
    b64 = img.data[0].b64_json
    out_path.write_bytes(base64.b64decode(b64))
    return out_path
