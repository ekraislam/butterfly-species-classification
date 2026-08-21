"""
App Utilities: Butterfly Species Metadata & Path Helpers.
Provides accurate, concise biological information for the 8 supported butterfly classes.
"""

import os

# Factual biological metadata for the 8 supported species
SPECIES_METADATA = {
    "MONARCH": {
        "scientific_name": "Danaus plexippus",
        "family": "Nymphalidae (Brush-footed Butterflies)",
        "appearance": "Tawny orange wings with bold black veins and two rows of white spots along black margins.",
        "distribution": "North & South America, introduced to Australia, Pacific Islands, and Western Europe.",
        "key_features": "Famous for spectacular multi-generational long-distance annual migrations and milkweed-derived toxicity."
    },
    "PAPER KITE": {
        "scientific_name": "Idea leuconoe",
        "family": "Nymphalidae (Danainae)",
        "appearance": "Large, delicate translucent ivory-white wings patterned with prominent black veins and scattered spots.",
        "distribution": "Southeast Asia, Philippines, Taiwan, and southern Japan.",
        "key_features": "Also known as the Rice Paper or Large Tree Nymph; famous for slow, buoyant gliding flight."
    },
    "RED POSTMAN": {
        "scientific_name": "Heliconius erato",
        "family": "Nymphalidae (Heliconiinae)",
        "appearance": "Velvety black elongated forewings with vivid crimson-red vertical bands and yellow ventral markings.",
        "distribution": "Central and South America, from Mexico through the Amazon basin.",
        "key_features": "Classic example of Müllerian mimicry alongside Heliconius melpomene; long-lived pollen-feeding species."
    },
    "ADONIS": {
        "scientific_name": "Polyommatus bellargus",
        "family": "Lycaenidae (Gossamer-winged Butterflies)",
        "appearance": "Males have brilliant electric-blue upperwings with white-checkered fringe; females are brown with blue dusting.",
        "distribution": "Southern & Central Europe, southern England.",
        "key_features": "Specialist of warm limestone and chalk grasslands; caterpillars share symbiotic relationships with ants."
    },
    "GREEN CELLED CATTLEHEART": {
        "scientific_name": "Parides sesostris",
        "family": "Papilionidae (Swallowtails)",
        "appearance": "Deep velvety black wings; males feature bright emerald-green forewing patches and crimson-red ventral body spots.",
        "distribution": "Tropical rainforests from Mexico to the Amazon Basin in Brazil.",
        "key_features": "Shade-tolerant forest swallowtail; toxic to predators due to aristolochic acids from larval host plants."
    },
    "SOUTHERN DOGFACE": {
        "scientific_name": "Zerene cesonia",
        "family": "Pieridae (Whites and Yellows)",
        "appearance": "Vibrant sulfur-yellow wings with a distinctive dark silhouette on each forewing resembling a dog's profile.",
        "distribution": "Southern United States through Central America to northern South America.",
        "key_features": "Fast, erratic flyer inhabiting open fields, pastures, and desert scrub; larvae feed on legumes."
    },
    "ORANGE OAKLEAF": {
        "scientific_name": "Kallima inachus",
        "family": "Nymphalidae (Nymphalinae)",
        "appearance": "Dorsal wings show rich royal blue and bright orange bands; ventral wings display master-class dead-leaf camouflage.",
        "distribution": "Tropical Asia from India and Nepal eastward to China, Taiwan, and Indochina.",
        "key_features": "One of nature's most iconic camouflage examples; completely resembles a dry leaf with midribs when wings are closed."
    },
    "CLODIUS PARNASSIAN": {
        "scientific_name": "Parnassius clodius",
        "family": "Papilionidae (Parnassiinae)",
        "appearance": "Semi-translucent milky-white wings dusted with dark charcoal scales and prominent red ocelli (eye-spots) ringed in black.",
        "distribution": "Montane Western North America, from Alaska through the Pacific Northwest and northern California.",
        "key_features": "Cold-adapted alpine and subalpine species; related to swallowtails but lacks tails, flying in mountain meadows."
    }
}

def resolve_project_paths():
    """
    Returns absolute paths to models, prepared_dataset, and root directory,
    ensuring stability whether the app is launched from the workspace root or inside app/.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    checkpoint_path = os.path.join(root_dir, "models", "butterfly_resnet18_best.pth")
    test_data_dir = os.path.join(root_dir, "prepared_dataset", "test")

    return {
        "root_dir": root_dir,
        "checkpoint_path": checkpoint_path,
        "test_data_dir": test_data_dir
    }
