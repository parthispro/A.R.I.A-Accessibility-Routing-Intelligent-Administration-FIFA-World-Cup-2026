"""Deterministic offline fallback assistant.

Runs when no Gemini API key is configured or the live API fails. Routes the
user's message to an intent via per-language keyword tables (en/es/fr/ar),
calls the same tool functions the live assistant uses (``app.tools``), and
renders the result through language-specific templates.

Pure stdlib, deterministic, no network. Output is screen-reader-friendly:
short plain sentences, no emoji or ASCII art.
"""

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from app import data, tools

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "ar", "hi")

#: Intents checked in priority order; first keyword match wins.
_INTENT_PRIORITY: tuple[str, ...] = (
    "accessibility",
    "services",
    "schedule",
    "food_water",
    "navigation",
    "greeting",
)

#: Keyword tables per intent and language (lowercase, accents stripped).
_INTENT_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "accessibility": {
        "en": (
            "wheelchair", "accessible", "accessibility", "disability",
            "disabled", "sensory", "autism", "autistic", "quiet", "quietest",
            "hearing", "deaf", "blind", "vision", "sight", "braille",
            "elevator", "lift", "ramp", "step-free", "mobility", "restroom",
            "toilet", "sign language", "listening",
        ),
        "es": (
            "silla de ruedas", "accesible", "accesibilidad", "discapacidad",
            "sensorial", "autismo", "sala tranquila", "audicion", "auditiva",
            "sordo", "sorda", "ciego", "ciega", "vision", "braille",
            "ascensor", "elevador", "rampa", "movilidad", "bano", "banos",
        ),
        "fr": (
            "fauteuil roulant", "accessible", "accessibilite", "handicap",
            "sensoriel", "sensorielle", "autisme", "salle calme", "audition",
            "auditif", "sourd", "sourde", "aveugle", "malvoyant", "braille",
            "ascenseur", "rampe", "mobilite", "toilettes",
        ),
        # Both bare and article-prefixed forms are listed: Arabic attaches the
        # definite article "ال" directly to the noun, and a word boundary sits
        # between words, not between "ال" and its stem.
        "ar": (
            "كرسي متحرك", "الكرسي المتحرك", "كراسي متحركة", "امكانية الوصول",
            "اعاقة", "الاعاقة", "معاق", "المعاقين", "ذوي الاحتياجات الخاصة",
            "حسي", "حسية", "غرفة حسية", "توحد", "التوحد", "هادئ", "هادئة",
            "مكان هادئ", "سمع", "السمع", "ضعف السمع", "اصم", "الصم",
            "لغة الاشارة", "سماعة", "اعمى", "المكفوفين", "مكفوف", "بصر",
            "البصر", "برايل", "مصعد", "المصعد", "مصاعد", "المصاعد", "منحدر",
            "المنحدر", "مرحاض", "المرحاض", "دورة مياه", "دورة المياه",
            "دورات المياه", "حمام", "الحمام",
        ),
        "hi": (
            "व्हीलचेयर", "सुलभ", "सुगम्यता", "विकलांगता", "विकलांग", "संवेदी", 
            "ऑटिज़्म", "शांत", "सुनने", "बधिर", "अंधा", "दृष्टि", "ब्रेल",
            "लिफ्ट", "रैंप", "गतिशीलता", "शौचालय", "सांकेतिक भाषा",
        ),
    },
    "services": {
        "en": (
            "nursing", "breastfeed", "lactation", "baby", "first aid",
            "medical", "medic", "prayer", "pray", "multi-faith",
        ),
        "es": (
            "lactancia", "amamantar", "bebe", "primeros auxilios",
            "enfermeria", "oracion", "rezar",
        ),
        "fr": (
            "allaitement", "allaiter", "bebe", "premiers secours",
            "infirmerie", "priere", "prier",
        ),
        "ar": (
            "رضاعة", "الرضاعة", "ارضاع", "غرفة رضاعة", "رضيع", "حليب",
            "اسعاف", "الاسعاف", "اسعافات", "اسعافات اولية", "طبي", "طبيب",
            "رعاية طبية", "صلاة", "الصلاة", "مصلى", "دعاء", "متعدد الاديان",
        ),
        "hi": (
            "नर्सिंग", "स्तनपान", "बच्चा", "प्राथमिक चिकित्सा", "चिकित्सा",
            "प्रार्थना", "बहु-धर्म", "चिकित्सक",
        ),
    },
    "schedule": {
        "en": (
            "match", "opening", "final", "kickoff", "kick-off", "schedule",
            "fixture", "when is", "what time", "ticket",
        ),
        "es": (
            "partido", "inauguracion", "inaugural", "final", "calendario",
            "horario", "cuando", "boleto", "entradas",
        ),
        "fr": (
            "match", "ouverture", "finale", "calendrier", "horaire", "quand",
            "coup d'envoi", "billet",
        ),
        "ar": (
            "مباراة", "المباراة", "مباريات", "المباريات", "افتتاح", "الافتتاح",
            "المباراة الافتتاحية", "افتتاحية", "نهائي", "النهائي",
            "المباراة النهائية", "نهائية", "جدول", "الجدول", "الجدول الزمني",
            "موعد", "المواعيد", "متى", "توقيت", "تذكرة", "التذكرة", "تذاكر",
            "التذاكر", "بطاقة",
        ),
        "hi": (
            "मैच", "उद्घाटन", "फाइनल", "किकऑफ", "अनुसूची", "शेड्यूल", "कार्यक्रम",
            "कब है", "कितने बजे", "टिकट",
        ),
    },
    "food_water": {
        "en": ("water", "food", "drink", "eat", "thirsty", "hungry", "snack",
               "concession"),
        "es": ("agua", "comida", "comer", "beber", "sed", "hambre"),
        "fr": ("eau", "nourriture", "manger", "boire", "soif", "faim"),
        "ar": (
            "ماء", "الماء", "مياه", "المياه", "مياه شرب", "طعام", "الطعام",
            "اكل", "الاكل", "ماكولات", "شراب", "مشروب", "مشروبات", "عطش",
            "عطشان", "جوع", "جائع", "وجبة",
        ),
        "hi": (
            "पानी", "भोजन", "खाना", "पीना", "प्यास", "भूख", "स्नैक", "जल",
        ),
    },
    "navigation": {
        "en": (
            "gate", "entrance", "entry", "route", "way to", "seat", "section",
            "how do i get", "where is", "directions", "navigate",
        ),
        "es": (
            "puerta", "entrada", "ruta", "asiento", "seccion", "como llego",
            "donde esta", "donde queda", "acceso",
        ),
        "fr": (
            "porte", "entree", "itineraire", "chemin", "siege", "section",
            "comment aller", "ou est", "ou sont", "acces",
        ),
        "ar": (
            "بوابة", "البوابة", "بوابات", "البوابات", "مدخل", "المدخل",
            "مداخل", "دخول", "مسار", "المسار", "طريق", "الطريق", "طريقي",
            "مقعد", "المقعد", "مقعدي", "قسم", "القسم", "منطقة", "كيف اصل",
            "كيف اذهب", "اين", "اتجاهات", "الاتجاهات", "خريطة",
        ),
        "hi": (
            "गेट", "प्रवेश", "मार्ग", "रास्ता", "सीट", "अनुभाग", "मैं कैसे पहुँचूँ",
            "कहाँ है", "दिशा", "नेविगेट", "द्वार",
        ),
    },
    "greeting": {
        "en": ("hello", "hi", "hey", "good morning", "good afternoon",
               "good evening", "help"),
        "es": ("hola", "buenos dias", "buenas tardes", "buenas noches",
               "ayuda"),
        "fr": ("bonjour", "salut", "bonsoir", "aide"),
        "ar": (
            "مرحبا", "السلام عليكم", "السلام", "سلام", "اهلا", "اهلا وسهلا",
            "صباح الخير", "مساء الخير", "مساعدة", "ساعدني", "هلا",
        ),
        "hi": (
            "नमस्ते", "हैलो", "सुप्रभात", "शुभ संध्या", "मदद", "सहायता",
        ),
    },
}

#: Keywords (all languages mixed) that pick a specific accessibility need.
_NEED_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mobility", (
        "wheelchair", "silla de ruedas", "fauteuil", "elevator", "lift",
        "ascensor", "elevador", "ascenseur", "ramp", "rampa", "rampe",
        "step-free", "mobility", "movilidad", "mobilite", "restroom",
        "toilet", "bano", "banos", "toilettes", "seating", "asiento",
        "كرسي متحرك", "الكرسي المتحرك", "مصعد", "المصعد", "مصاعد", "منحدر",
        "المنحدر", "مرحاض", "المرحاض", "دورة مياه", "دورة المياه", "حمام",
        "الحمام", "مقعد", "المقعد", "التنقل", "حركة",
        "व्हीलचेयर", "लिफ्ट", "रैंप", "गतिशीलता", "शौचालय", "सीट",
    )),
    ("hearing", (
        "hearing", "deaf", "sordo", "sorda", "auditiva", "audicion", "sourd",
        "sourde", "auditif", "audition", "listening", "caption",
        "sign language",
        "سمع", "السمع", "ضعف السمع", "اصم", "الصم", "لغة الاشارة", "سماعة",
        "सुनने", "बधिर", "सांकेतिक भाषा",
    )),
    ("vision", (
        "blind", "vision", "sight", "braille", "ciego", "ciega", "aveugle",
        "malvoyant", "vue",
        "اعمى", "المكفوفين", "مكفوف", "بصر", "البصر", "برايل", "ضعف البصر",
        "अंधा", "दृष्टि", "ब्रेल",
    )),
    ("sensory", (
        "sensory", "sensorial", "sensoriel", "sensorielle", "autism",
        "autistic", "autismo", "autisme", "quiet", "quietest", "noise",
        "ruido", "bruit", "tranquila", "calme",
        "حسي", "حسية", "توحد", "التوحد", "هادئ", "هادئة", "ضوضاء", "ضجيج",
        "غرفة حسية",
        "संवेदी", "ऑटिज़्म", "शांत", "शोर",
    )),
)

#: Keywords picking a specific venue service within the services intent.
_SERVICE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nursing_room", ("nursing", "breastfeed", "lactation", "lactancia",
                      "amamantar", "allaitement", "allaiter", "baby", "bebe",
                      "رضاعة", "الرضاعة", "ارضاع", "رضيع", "حليب",
                      "नर्सिंग", "स्तनपान", "बच्चा")),
    ("first_aid", ("first aid", "medical", "medic", "primeros auxilios",
                   "enfermeria", "premiers secours", "infirmerie",
                   "اسعاف", "الاسعاف", "اسعافات", "طبي", "طبيب", "رعاية طبية",
                   "प्राथमिक चिकित्सा", "चिकित्सा", "चिकित्सक")),
    ("prayer_room", ("prayer", "pray", "oracion", "rezar", "priere", "prier",
                     "multi-faith",
                     "صلاة", "الصلاة", "مصلى", "دعاء", "متعدد الاديان",
                     "प्रार्थना", "बहु-धर्म")),
)

#: Human-readable congestion levels per language.
_LEVELS: dict[str, dict[str, str]] = {
    "en": {"low": "low", "moderate": "moderate", "high": "high"},
    "es": {"low": "baja", "moderate": "moderada", "high": "alta"},
    "fr": {"low": "faible", "moderate": "moderee", "high": "elevee"},
    "ar": {"low": "منخفض", "moderate": "متوسط", "high": "مرتفع"},
    "hi": {"low": "कम", "moderate": "मध्यम", "high": "अधिक"},
}

#: Labels for accessibility fields per language.
_FIELD_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "gates": "Accessible gates",
        "accessible_seating": "Seating",
        "sensory_room": "Sensory support",
        "assistive_listening": "Assistive listening",
        "vision_support": "Vision support",
        "elevators": "Elevators",
        "accessible_restrooms": "Accessible restrooms",
        "quiet_route_hint": "Quiet route",
    },
    "es": {
        "gates": "Puertas accesibles",
        "accessible_seating": "Asientos",
        "sensory_room": "Apoyo sensorial",
        "assistive_listening": "Audición asistida",
        "vision_support": "Apoyo visual",
        "elevators": "Ascensores",
        "accessible_restrooms": "Baños accesibles",
        "quiet_route_hint": "Ruta tranquila",
    },
    "fr": {
        "gates": "Portes accessibles",
        "accessible_seating": "Places",
        "sensory_room": "Soutien sensoriel",
        "assistive_listening": "Aide à l'écoute",
        "vision_support": "Assistance visuelle",
        "elevators": "Ascenseurs",
        "accessible_restrooms": "Toilettes accessibles",
        "quiet_route_hint": "Itinéraire calme",
    },
    "ar": {
        "gates": "البوابات المتاحة",
        "accessible_seating": "المقاعد",
        "sensory_room": "الدعم الحسي",
        "assistive_listening": "أجهزة الاستماع المساعدة",
        "vision_support": "دعم البصر",
        "elevators": "المصاعد",
        "accessible_restrooms": "دورات المياه المتاحة",
        "quiet_route_hint": "المسار الهادئ",
    },
    "hi": {
        "gates": "सुलभ प्रवेश द्वार",
        "accessible_seating": "बैठने की व्यवस्था",
        "sensory_room": "संवेदी समर्थन",
        "assistive_listening": "सहायक श्रवण",
        "vision_support": "दृष्टि समर्थन",
        "elevators": "लिफ्ट",
        "accessible_restrooms": "सुलभ शौचालय",
        "quiet_route_hint": "शांत मार्ग",
    },
}

#: Response templates per language. Values are full translations, not
#: prefixed English. Dataset facts are interpolated as-is.
_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "greeting": (
            "Hello. I am A.R.I.A, your accessibility assistant for the "
            "FIFA World Cup 2026. You can ask about accessible gates, "
            "seating, sensory rooms, water, or match dates."
        ),
        "help": (
            "I can help with accessibility services, gates and routes, food "
            "and water, and the match schedule. Try asking: which gate has "
            "wheelchair access?"
        ),
        "pick_venue": (
            "Please choose a stadium first so I can give exact information. "
            "For example: {examples}."
        ),
        "unverified": (
            "Please note: these details are not yet verified with the venue."
        ),
        "acc_intro": "Accessibility at {venue}.",
        "nursing_room": "Yes. {venue} has a nursing room. Location: {value}.",
        "first_aid": "First aid at {venue}: {value}.",
        "prayer_room": "Prayer space at {venue}: {value}.",
        "food_water": (
            "Water at {venue}: {water}. Food stands are on every concourse. "
            "Staff can point you to accessible counters."
        ),
        "navigation": (
            "Recommended entrance at {venue}: {gate}. {notes}. Current "
            "congestion is {level} (simulated live data). Tip: {hint}"
        ),
        "outage": "Warning: the elevator near {gate} is out of service.",
        "schedule_opening": "{venue} hosts the opening match on {date}.",
        "schedule_final": "{venue} hosts the final on {date}.",
        "schedule_general": (
            "The opening match is on {open_date} at {open_venue}. "
            "The final is on {final_date} at {final_venue}."
        ),
        "tickets": (
            "FIFA offers three accessibility ticket types: {types}."
        ),
    },
    "es": {
        "greeting": (
            "Hola. Soy A.R.I.A, su asistente de accesibilidad para la "
            "Copa Mundial de la FIFA 2026. Puede preguntarme por puertas "
            "accesibles, asientos, salas sensoriales, agua o fechas de "
            "partidos."
        ),
        "help": (
            "Puedo ayudarle con servicios de accesibilidad, puertas y rutas, "
            "comida y agua, y el calendario de partidos. Pruebe a preguntar: "
            "¿qué puerta tiene acceso para silla de ruedas?"
        ),
        "pick_venue": (
            "Por favor, elija primero un estadio para poder darle "
            "información exacta. Por ejemplo: {examples}."
        ),
        "unverified": (
            "Tenga en cuenta: estos datos aún no están verificados con el "
            "estadio."
        ),
        "acc_intro": "Accesibilidad en {venue}.",
        "nursing_room": (
            "Sí. {venue} tiene una sala de lactancia. Ubicación: {value}."
        ),
        "first_aid": "Primeros auxilios en {venue}: {value}.",
        "prayer_room": "Sala de oración en {venue}: {value}.",
        "food_water": (
            "Agua en {venue}: {water}. Hay puestos de comida en todos los "
            "pasillos. El personal puede indicarle los mostradores "
            "accesibles."
        ),
        "navigation": (
            "Entrada recomendada en {venue}: {gate}. {notes}. La congestión "
            "actual es {level} (datos simulados). Consejo: {hint}"
        ),
        "outage": (
            "Aviso: el ascensor cerca de {gate} está fuera de servicio."
        ),
        "schedule_opening": "{venue} acoge el partido inaugural el {date}.",
        "schedule_final": "{venue} acoge la final el {date}.",
        "schedule_general": (
            "El partido inaugural es el {open_date} en {open_venue}. "
            "La final es el {final_date} en {final_venue}."
        ),
        "tickets": (
            "La FIFA ofrece tres tipos de entradas de accesibilidad: {types}."
        ),
    },
    "fr": {
        "greeting": (
            "Bonjour. Je suis A.R.I.A, votre assistant d'accessibilité "
            "pour la Coupe du Monde de la FIFA 2026. Vous pouvez me poser "
            "des questions sur les portes accessibles, les places, les "
            "salles sensorielles, l'eau ou les dates des matchs."
        ),
        "help": (
            "Je peux vous aider avec les services d'accessibilité, les "
            "portes et itinéraires, la nourriture et l'eau, et le calendrier "
            "des matchs. Essayez de demander : quelle porte est accessible "
            "en fauteuil roulant ?"
        ),
        "pick_venue": (
            "Veuillez d'abord choisir un stade pour que je puisse donner "
            "des informations exactes. Par exemple : {examples}."
        ),
        "unverified": (
            "Veuillez noter : ces informations ne sont pas encore vérifiées "
            "auprès du stade."
        ),
        "acc_intro": "Accessibilité à {venue}.",
        "nursing_room": (
            "Oui. {venue} dispose d'une salle d'allaitement. "
            "Emplacement : {value}."
        ),
        "first_aid": "Premiers secours à {venue} : {value}.",
        "prayer_room": "Salle de prière à {venue} : {value}.",
        "food_water": (
            "Eau à {venue} : {water}. Des stands de nourriture se trouvent "
            "sur toutes les coursives. Le personnel peut vous indiquer les "
            "comptoirs accessibles."
        ),
        "navigation": (
            "Entrée recommandée à {venue} : {gate}. {notes}. La congestion "
            "actuelle est {level} (données simulées). Conseil : {hint}"
        ),
        "outage": (
            "Attention : l'ascenseur près de {gate} est hors service."
        ),
        "schedule_opening": "{venue} accueille le match d'ouverture le {date}.",
        "schedule_final": "{venue} accueille la finale le {date}.",
        "schedule_general": (
            "Le match d'ouverture a lieu le {open_date} à {open_venue}. "
            "La finale a lieu le {final_date} à {final_venue}."
        ),
        "tickets": (
            "La FIFA propose trois types de billets d'accessibilité : "
            "{types}."
        ),
    },
    "ar": {
        "greeting": (
            "مرحباً. أنا A.R.I.A، مساعدك لإمكانية الوصول في كأس العالم "
            "FIFA 2026. يمكنك أن تسألني عن البوابات المتاحة، والمقاعد، والغرف "
            "الحسية، والماء، أو مواعيد المباريات."
        ),
        "help": (
            "يمكنني المساعدة في خدمات إمكانية الوصول، والبوابات والمسارات، "
            "والطعام والماء، وجدول المباريات. جرّب أن تسأل: أي بوابة متاحة "
            "للكرسي المتحرك؟"
        ),
        "pick_venue": (
            "من فضلك اختر ملعباً أولاً حتى أتمكن من إعطائك معلومات دقيقة. "
            "على سبيل المثال: {examples}."
        ),
        "unverified": (
            "يرجى الملاحظة: هذه التفاصيل لم يتم التحقق منها بعد مع الملعب."
        ),
        "acc_intro": "إمكانية الوصول في {venue}.",
        "nursing_room": "نعم. يوجد في {venue} غرفة رضاعة. الموقع: {value}.",
        "first_aid": "الإسعافات الأولية في {venue}: {value}.",
        "prayer_room": "مكان الصلاة في {venue}: {value}.",
        "food_water": (
            "الماء في {venue}: {water}. تتوفر أكشاك الطعام في جميع الممرات. "
            "يمكن للموظفين إرشادك إلى المنافذ المتاحة."
        ),
        "navigation": (
            "المدخل الموصى به في {venue}: {gate}. {notes}. الازدحام الحالي "
            "{level} (بيانات محاكاة). نصيحة: {hint}"
        ),
        "outage": "تحذير: المصعد القريب من {gate} خارج الخدمة.",
        "schedule_opening": "يستضيف {venue} مباراة الافتتاح في {date}.",
        "schedule_final": "يستضيف {venue} المباراة النهائية في {date}.",
        "schedule_general": (
            "تقام مباراة الافتتاح في {open_date} في {open_venue}. وتقام "
            "المباراة النهائية في {final_date} في {final_venue}."
        ),
        "tickets": (
            "تقدم FIFA ثلاثة أنواع من تذاكر إمكانية الوصول: {types}."
        ),
    },
    "hi": {
        "greeting": (
            "नमस्ते। मैं A.R.I.A हूँ, फीफा विश्व कप 2026 के लिए आपका सुगम्यता "
            "सहायक। आप सुलभ प्रवेश द्वार, बैठने की जगह, संवेदी कमरे, पानी, "
            "या मैच की तारीखों के बारे में पूछ सकते हैं।"
        ),
        "help": (
            "मैं सुगम्यता सेवाओं, प्रवेश द्वारों और मार्गों, भोजन और पानी, "
            "और मैच के कार्यक्रम में मदद कर सकता हूँ। पूछने का प्रयास करें: "
            "किस गेट पर व्हीलचेयर की सुविधा है?"
        ),
        "pick_venue": (
            "कृपया पहले एक स्टेडियम चुनें ताकि मैं सटीक जानकारी दे सकूँ। "
            "उदाहरण के लिए: {examples}।"
        ),
        "unverified": (
            "कृपया ध्यान दें: इन विवरणों की अभी तक स्टेडियम के साथ पुष्टि नहीं हुई है।"
        ),
        "acc_intro": "{venue} में सुगम्यता।",
        "nursing_room": "हाँ। {venue} में एक नर्सिंग रूम है। स्थान: {value}।",
        "first_aid": "{venue} में प्राथमिक चिकित्सा: {value}।",
        "prayer_room": "{venue} में प्रार्थना कक्ष: {value}।",
        "food_water": (
            "{venue} में पानी: {water}। भोजन के स्टॉल हर रास्ते पर हैं। "
            "कर्मचारी आपको सुलभ काउंटरों की ओर इशारा कर सकते हैं।"
        ),
        "navigation": (
            "{venue} में अनुशंसित प्रवेश द्वार: {gate}। {notes}। वर्तमान भीड़ "
            "{level} है (सिम्युलेटेड डेटा)। सुझाव: {hint}"
        ),
        "outage": "चेतावनी: {gate} के पास वाली लिफ्ट सेवा से बाहर है।",
        "schedule_opening": "{venue} {date} को उद्घाटन मैच की मेजबानी करेगा।",
        "schedule_final": "{venue} {date} को फाइनल की मेजबानी करेगा।",
        "schedule_general": (
            "उद्घाटन मैच {open_date} को {open_venue} में है। "
            "फाइनल {final_date} को {final_venue} में है।"
        ),
        "tickets": (
            "फीफा तीन प्रकार के सुगम्यता टिकट प्रदान करता है: {types}।"
        ),
    },
}

def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'visión' matches 'vision'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


#: Arabic single-letter proclitics (wa/fa/bi/ka/li = and/then/with/like/for)
#: attach directly to the following word, so a plain word boundary misses them
#: (e.g. "بالتوحد" = with-the-autism). Allowing a short leading cluster lets the
#: keyword "التوحد"/"توحد" still match. The class holds only Arabic letters, so
#: it consumes nothing in en/es/fr text and leaves those matches unchanged.
_AR_PROCLITICS = "وفبكل"


def _contains(normalized: str, keyword: str) -> bool:
    """Whole-word (or whole-phrase) match against the normalized message.

    The keyword is normalized with the same pipeline as the message so keyword
    tables can be authored in natural orthography. This matters for Arabic,
    where NFKD folds hamza-carrying letters (``أ إ آ`` -> ``ا``, ``ئ`` -> ``ي``)
    and strips diacritics — a raw keyword like ``النهائي`` would otherwise never
    match its own normalized form. It is a no-op for the already-plain en/es/fr
    keywords.
    """
    kw = re.escape(_normalize(keyword))
    return re.search(rf"\b[{_AR_PROCLITICS}]{{0,2}}{kw}\b", normalized) is not None


def _detect_intent(normalized: str) -> tuple[str, str | None]:
    """Return (intent, language-of-matched-keyword). Fallback: ('help', None)."""
    for intent in _INTENT_PRIORITY:
        for lang in SUPPORTED_LANGUAGES:
            for keyword in _INTENT_KEYWORDS[intent][lang]:
                if _contains(normalized, keyword):
                    return intent, lang
    return "help", None


def _resolve_language(code: object, detected: str | None) -> str:
    """Profile language wins; unknown codes fall back to English."""
    if isinstance(code, str) and code.strip():
        short = code.strip().lower()[:2]
        return short if short in _TEMPLATES else "en"
    return detected if detected in _TEMPLATES else "en"


def _resolve_need(normalized: str, profile: Mapping[str, Any]) -> str:
    """Pick the accessibility need from message keywords, then profile needs."""
    for need, keywords in _NEED_KEYWORDS:
        if any(_contains(normalized, keyword) for keyword in keywords):
            return need
    for raw in profile.get("needs") or []:
        if isinstance(raw, str) and raw.lower() in tools.VALID_NEEDS:
            return raw.lower()
    return "general"


def _sentence(text: str) -> str:
    """Ensure the fragment ends with sentence punctuation."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _clause(text: str) -> str:
    """Strip a trailing period so the fragment can sit inside a template."""
    return text.strip().rstrip(".")


def _venue_examples() -> str:
    """Two-three example venues for the 'pick a venue' prompt."""
    return ", ".join(
        f"{venue['name']} ({venue['id']})" for venue in data.list_venues()[:3]
    )


def _accessibility_answer(venue_id: str, need: str, lang: str) -> str:
    """Render the need-filtered accessibility facts for a venue."""
    result = tools.find_accessible_services(venue_id, need=need)
    templates, labels = _TEMPLATES[lang], _FIELD_LABELS[lang]
    parts = [templates["acc_intro"].format(venue=result["venue_name"])]
    for field, value in result["services"].items():
        if field == "gates":
            names = ", ".join(g["name"] for g in value if g["accessible"])
            parts.append(f"{labels['gates']}: {names}.")
        else:
            parts.append(f"{labels[field]}: {_sentence(value)}")
    if not result["verified"]:
        parts.append(templates["unverified"])
    return " ".join(parts)


def _services_answer(
    venue: data.Venue, normalized: str, lang: str,
) -> str:
    """Answer nursing room / first aid / prayer room questions."""
    templates = _TEMPLATES[lang]
    services = venue["services"]
    for service, keywords in _SERVICE_KEYWORDS:
        if any(_contains(normalized, keyword) for keyword in keywords):
            return templates[service].format(
                venue=venue["name"], value=_clause(services[service]),
            )
    return " ".join(
        templates[service].format(
            venue=venue["name"], value=_clause(services[service]),
        )
        for service, _ in _SERVICE_KEYWORDS
    )


def _navigation_answer(venue_id: str, lang: str) -> str:
    """Recommend the current quietest accessible entrance (simulated feed)."""
    templates = _TEMPLATES[lang]
    status = tools.get_live_status(venue_id)
    info = tools.get_venue_info(venue_id)
    gate_name = status["quiet_entrance"]
    gate: dict[str, Any] = next(
        (g for g in info["gates"] if g["name"] == gate_name), {},
    )
    level = next(
        (
            entry["congestion"]
            for entry in status["gate_congestion"]
            if entry["gate"] == gate_name
        ),
        "low",
    )
    hint = tools.find_accessible_services(venue_id, need="sensory")
    parts = [
        templates["navigation"].format(
            venue=info["name"],
            gate=gate_name,
            notes=_clause(gate.get("notes", "")),
            level=_LEVELS[lang][level],
            hint=_sentence(hint["services"]["quiet_route_hint"]),
        ),
    ]
    if status["elevator_outage"] is not None:
        parts.append(
            templates["outage"].format(gate=status["elevator_outage"]["gate"]),
        )
    return " ".join(parts)


def _schedule_answer(venue: data.Venue | None, lang: str) -> str:
    """Render opening match / final dates, venue-specific when possible."""
    templates = _TEMPLATES[lang]
    parts: list[str] = []
    if venue is not None:
        matchday = tools.get_venue_info(venue["id"])["matchday"]
        if "hosts_opening_match" in matchday:
            parts.append(
                templates["schedule_opening"].format(
                    venue=venue["name"], date=matchday["hosts_opening_match"],
                ),
            )
        if "hosts_final" in matchday:
            parts.append(
                templates["schedule_final"].format(
                    venue=venue["name"], date=matchday["hosts_final"],
                ),
            )
    tournament = data.load_venues()["tournament"]
    if not parts:
        opening, final = tournament["openingMatch"], tournament["final"]
        opening_venue = data.get_venue(opening["venueId"])
        final_venue = data.get_venue(final["venueId"])
        parts.append(
            templates["schedule_general"].format(
                open_date=opening["date"],
                open_venue=opening_venue["name"] if opening_venue else "?",
                final_date=final["date"],
                final_venue=final_venue["name"] if final_venue else "?",
            ),
        )
    parts.append(
        templates["tickets"].format(
            types=", ".join(tournament["accessibility_tickets"]["types"]),
        ),
    )
    return " ".join(parts)


def _venue_intent_answer(
    intent: str, venue: data.Venue, normalized: str, profile: Mapping[str, Any], lang: str,
) -> str:
    """Render the reply for an intent that requires a selected venue."""
    templates = _TEMPLATES[lang]
    if intent == "accessibility":
        return _accessibility_answer(
            venue["id"], _resolve_need(normalized, profile), lang,
        )
    if intent == "services":
        return _services_answer(venue, normalized, lang)
    if intent == "food_water":
        return templates["food_water"].format(
            venue=venue["name"], water=_clause(venue["services"]["water"]),
        )
    return _navigation_answer(venue["id"], lang)


def offline_answer(message: str, profile: Mapping[str, Any] | None = None) -> str:
    """Answer ``message`` deterministically without any LLM or network.

    ``profile`` may carry ``language`` (2-letter code), ``needs`` (list of
    need values), and ``venue_id``. Unknown languages fall back to English;
    if no venue is selected and the intent requires one, the reply asks the
    user to pick a venue.
    """
    profile = profile or {}
    normalized = _normalize(message or "")
    intent, detected_lang = _detect_intent(normalized)
    lang = _resolve_language(profile.get("language"), detected_lang)
    templates = _TEMPLATES[lang]

    if intent == "greeting":
        return templates["greeting"]
    if intent == "help":
        return templates["help"]

    venue_id = profile.get("venue_id")
    venue = data.get_venue(venue_id) if isinstance(venue_id, str) else None

    if intent == "schedule":
        return _schedule_answer(venue, lang)
    # Every remaining intent needs a venue.
    if venue is None:
        return templates["pick_venue"].format(examples=_venue_examples())
    return _venue_intent_answer(intent, venue, normalized, profile, lang)
