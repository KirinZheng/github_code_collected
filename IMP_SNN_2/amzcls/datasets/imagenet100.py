from mmpretrain.datasets import ImageNet

from amzcls.registry import DATASETS

IMAGENET100_INDICES = {
    "n01968897": "chambered nautilus, pearly nautilus, nautilus",
    "n01770081": "harvestman, daddy longlegs, Phalangium opilio", "n01818515": "macaw", "n02011460": "bittern",
    "n01496331": "electric ray, crampfish, numbfish, torpedo", "n01847000": "drake", "n01687978": "agama",
    "n01740131": "night snake, Hypsiglena torquata",
    "n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "n01491361": "tiger shark, Galeocerdo cuvieri", "n02007558": "flamingo", "n01735189": "garter snake, grass snake",
    "n01630670": "common newt, Triturus vulgaris", "n01440764": "tench, Tinca tinca",
    "n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "n02002556": "white stork, Ciconia ciconia", "n01667778": "terrapin",
    "n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus", "n01924916": "flatworm, platyhelminth",
    "n01751748": "sea snake", "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "n01729977": "green snake, grass snake", "n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus",
    "n01608432": "kite", "n01443537": "goldfish, Carassius auratus", "n01770393": "scorpion", "n01855672": "goose",
    "n01560419": "bulbul", "n01592084": "chickadee", "n01914609": "sea anemone, anemone", "n01582220": "magpie",
    "n01667114": "mud turtle", "n01985128": "crayfish, crawfish, crawdad, crawdaddy", "n01820546": "lorikeet",
    "n01773797": "garden spider, Aranea diademata", "n02006656": "spoonbill", "n01986214": "hermit crab",
    "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "n01749939": "green mamba", "n01828970": "bee eater", "n02018795": "bustard",
    "n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "n01729322": "hognose snake, puff adder, sand viper", "n01677366": "common iguana, iguana, Iguana iguana",
    "n01734418": "king snake, kingsnake", "n01843383": "toucan", "n01806143": "peacock",
    "n01773549": "barn spider, Araneus cavaticus", "n01775062": "wolf spider, hunting spider",
    "n01728572": "thunder snake, worm snake, Carphophis amoenus", "n01601694": "water ouzel, dipper",
    "n01978287": "Dungeness crab, Cancer magister", "n01930112": "nematode, nematode worm, roundworm",
    "n01739381": "vine snake", "n01883070": "wombat", "n01774384": "black widow, Latrodectus mactans",
    "n02037110": "oystercatcher, oyster catcher", "n01795545": "black grouse",
    "n02027492": "red-backed sandpiper, dunlin, Erolia alpina", "n01531178": "goldfinch, Carduelis carduelis",
    "n01944390": "snail", "n01494475": "hammerhead, hammerhead shark",
    "n01632458": "spotted salamander, Ambystoma maculatum",
    "n01698640": "American alligator, Alligator mississipiensis", "n01675722": "banded gecko",
    "n01877812": "wallaby, brush kangaroo", "n01622779": "great grey owl, great gray owl, Strix nebulosa",
    "n01910747": "jellyfish", "n01860187": "black swan, Cygnus atratus", "n01796340": "ptarmigan",
    "n01833805": "hummingbird", "n01685808": "whiptail, whiptail lizard",
    "n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes", "n01514859": "hen",
    "n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "n02058221": "albatross, mollymawk", "n01632777": "axolotl, mud puppy, Ambystoma mexicanum",
    "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "n01664065": "loggerhead, loggerhead turtle, Caretta caretta", "n02028035": "redshank, Tringa totanus",
    "n02012849": "crane", "n01776313": "tick", "n02077923": "sea lion", "n01774750": "tarantula",
    "n01742172": "boa constrictor, Constrictor constrictor", "n01943899": "conch",
    "n01798484": "prairie chicken, prairie grouse, prairie fowl", "n02051845": "pelican", "n01824575": "coucal",
    "n02013706": "limpkin, Aramus pictus", "n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "n01773157": "black and gold garden spider, Argiope aurantia",
    "n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea", "n01498041": "stingray",
    "n01978455": "rock crab, Cancer irroratus", "n01693334": "green lizard, Lacerta viridis",
    "n01950731": "sea slug, nudibranch", "n01829413": "hornbill", "n01514668": "cock"}

# IMAGENET100_CATEGORIES = (IMAGENET100_INDICES[index] for index in IMAGENET100_INDICES)
IMAGENET100_CATEGORIES = (
    'chambered nautilus, pearly nautilus, nautilus', 'harvestman, daddy longlegs, Phalangium opilio', 'macaw',
    'bittern', 'electric ray, crampfish, numbfish, torpedo', 'drake', 'agama', 'night snake, Hypsiglena torquata',
    'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'tiger shark, Galeocerdo cuvieri', 'flamingo',
    'garter snake, grass snake', 'common newt, Triturus vulgaris', 'tench, Tinca tinca',
    'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'white stork, Ciconia ciconia', 'terrapin',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'flatworm, platyhelminth', 'sea snake',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'green snake, grass snake',
    'bald eagle, American eagle, Haliaeetus leucocephalus', 'kite', 'goldfish, Carassius auratus', 'scorpion', 'goose',
    'bulbul', 'chickadee', 'sea anemone, anemone', 'magpie', 'mud turtle', 'crayfish, crawfish, crawdad, crawdaddy',
    'lorikeet', 'garden spider, Aranea diademata', 'spoonbill', 'hermit crab',
    'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'green mamba', 'bee eater',
    'bustard', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'hognose snake, puff adder, sand viper', 'common iguana, iguana, Iguana iguana', 'king snake, kingsnake', 'toucan',
    'peacock', 'barn spider, Araneus cavaticus', 'wolf spider, hunting spider',
    'thunder snake, worm snake, Carphophis amoenus', 'water ouzel, dipper', 'Dungeness crab, Cancer magister',
    'nematode, nematode worm, roundworm', 'vine snake', 'wombat', 'black widow, Latrodectus mactans',
    'oystercatcher, oyster catcher', 'black grouse', 'red-backed sandpiper, dunlin, Erolia alpina',
    'goldfinch, Carduelis carduelis', 'snail', 'hammerhead, hammerhead shark',
    'spotted salamander, Ambystoma maculatum', 'American alligator, Alligator mississipiensis', 'banded gecko',
    'wallaby, brush kangaroo', 'great grey owl, great gray owl, Strix nebulosa', 'jellyfish',
    'black swan, Cygnus atratus', 'ptarmigan', 'hummingbird', 'whiptail, whiptail lizard',
    'sidewinder, horned rattlesnake, Crotalus cerastes', 'hen',
    'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'albatross, mollymawk',
    'axolotl, mud puppy, Ambystoma mexicanum', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    'American coot, marsh hen, mud hen, water hen, Fulica americana', 'loggerhead, loggerhead turtle, Caretta caretta',
    'redshank, Tringa totanus', 'crane', 'tick', 'sea lion', 'tarantula', 'boa constrictor, Constrictor constrictor',
    'conch', 'prairie chicken, prairie grouse, prairie fowl', 'pelican', 'coucal', 'limpkin, Aramus pictus',
    'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'black and gold garden spider, Argiope aurantia',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'stingray', 'rock crab, Cancer irroratus',
    'green lizard, Lacerta viridis', 'sea slug, nudibranch', 'hornbill', 'cock')


@DATASETS.register_module()
class ImageNet100(ImageNet):
    METAINFO = {'classes': IMAGENET100_CATEGORIES}

    def __init__(self, *args, **kwargs):
        super(ImageNet100, self).__init__(*args, **kwargs)
