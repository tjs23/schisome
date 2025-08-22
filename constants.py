# General
UNKNOWN = 'unknown'
NAN = float('nan')
PROG_CHARS = ('#', '-')
READ_BUFFER = 2**16
# NumPy Zip format save keys
ARRAY_VALUES_TAG = 'arrydata_'
MARKER_CLASSES_TAG = 'mkrdata_'
MARKER_COLORS_TAG = 'mkrcolors_'
MARKER_LABELS_TAG = 'mkrclasses_'
PRED_CLASSES_TAG = 'predclassdata_'
PRED_LABELS_TAG  = 'predclasslabels_'
PROFILE_TAG = 'prfdata_'
PROJ_TAG = 'projdata_'
RAW_DATA_TAG = 'rawdata_'
# UniProt access
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/stream?query={}&format=tsv&fields={}'
UNIPROT_ALT_ID_URL = 'https://ftp.uniprot.org/pub/databases/uniprot/current%5Frelease/knowledgebase/complete/docs/sec%5Fac.txt'
# Plotting
COLOR_DICT = { 'RIBOSOME 40S':       '#000080',
               'RIBOSOME 60S':       '#800000',
               'CYTOSOLIC RIBOSOMES': '#800000',
	       'CHLOROPLAST':        '#50FF20',
               'CHLOROPLAST ENVELOPE': '#FF0000',
               'CIS-GOLGI':            '#00B0B0',
               'GOLGI CIS':            '#00B0B0',
               'MEDIAL GOLGI':         '#FFE000',
               'GOLGI MEDIAL':         '#FFE000',
               'MITOCHONDRIAL INNER MEMBRANE': '#FF4000',
               'MITOCHONDRIAL OUTER MEMBRANE': '#0080FF',
               'PLASTID; PEROXISOME':         '#FF80FF',
               'CHLOROPLAST ENVELOPE-OUTER':          '#FFA070',
               'CHLOROPLAST ENVELOPE-INNER':          '#FFFF80',
               'PLASTID ENVELOPE OUTER':          '#FFA070',
               'PLASTID ENVELOPE INNER':          '#FFFF80',
               'PLASTID ENVELOPE':          '#FF8000',
               'PLASTID NUCLEOID': '#FFFF40',
               'PLASTID RIBOSOME': '#0000FF',
               'PLASTID STROMA':   '#DDDD00',
               'PLASTOGLOBULES':   '#00FFFF',
               'PLASTID PLASTOGLOBULES':   '#00FFFF',
               'STROMA':           '#00FF00',
               'THYLAKOID':        '#FFD000',
               'PLASTID THYLAKOID':        '#FF00FF',
               'PLASTID MEMBRANE':        '#0080FF',
               'TRANS-GOLGI':      '#FF4000',
               'GOLGI TRANS':      '#FF4000',
               'ALTERNATIVE ENZYMES':      '#008000',
               'BIG COMPLEX MEMBER':       '#8000FF',
               'CALVINBB CYCLE':           '#FF00FF',
               'COMPLEX_1':                '#FF0000',
               'COMPLEX_1_SUBCOMPLEX':     '#800000',
               'COMPLEX_2':                '#FFD000',
               'COMPLEX_3':                '#008080',
               'COMPLEX_4':                '#0000FF',
               'COMPLEX_5':                '#00FFFF',
               'PSI':                      '#00FF00',
               'PSII':                     '#008000',
               'NUCLEUS-CHROMATIN':  '#905020',
               'CHROMATIN':  '#905020',
               'CENTROSOME':  '#FFFF00',
               'NUCLEAR MEMBRANE': '#00FFFF',
               'NUCLEAR LAMINA': '#00FFFF',
               'MIDBODY': '#00FF00',
               'NUCLEOLI': '#00B0B0',
               'ER/GOLGI':           '#CCCC00',
	       'TGN':                '#B080FF',
	       'VACUOLE':            '#00A000',
               'ENDOSOME':           '#00FFFF', #'#B080FF',
               'CYTOSKELETON':       '#008080',
               'CYTOSOL':            '#FF0000',
               'DUAL':               '#DDDDDD',
               'ER':                 '#0050FF',
               'ECM':                '#AADD00',#'#000000',
               'EXTRACELLULAR':      '#AADD00',#'#000000',
               'GOLGI':              '#FFD000',
               'LYSOSOME':           '#009000',
               'MITOCHONDRION':      '#FF7000',
               'MITOCHONDRIA':       '#FF7000',
               'MULTI-LOC':          '#DDDDDD',
               'NUCLEUS':            '#80B0F0',
               'PEROXISOME':         '#FF00FF',#'#FF80FF',
               'PLASTID':            '#50FF20',
               'PM':                 '#FF8888', # '#FF00FF',
               'PROTEASOME':         '#40D000',
               'RIBOSOMES':          '#7020FF',
               'RIBOSOMAL':          '#7020FF',
               'SHITE':              '#005050',#'#000000',
               'unknown':            '#404040',
               'UNKNOWN':            '#404040',
               'Selection':            '#FFFFFF',
               'm0': '#808080',
               'm1': '#404040',
               'm2': '#0080FF',
               'm3': '#FFFF00',
               'm4': '#FF0000',
               'm5': '#FF00FF',
               'CGN': '#00A0A0', # Golgi nuc
               'CEC': '#FFEEDD', # Cyt ext
               'CNR': '#A0FFA0', # Next to ribo
               'PMS': '#40FFFF', # PM sub pop
               'PLASMODESMATA': '#FFFF80', 
               'G1': '#00A0A0', 
               'G2': '#EEDDCC', 
               'G3': '#A0FFA0', 
                   } # '#C0C0C0'}
# SQL database                   
DB_ORGANELLE_CONV_DICT = {'MITOCHONDRIAL INNER MEMBRANE': 'MIM',
                         'MITOCHONDRIAL OUTER MEMBRANE': 'MOM',
                         'MITOCHONDRIA': 'MITOCHONDRION',
                         }

DB_ORGANELLE_INFO = {'CEC':              ('Cytosol-ECM cluster','#FFEEDD'), # Cyt ext
                     'CGN':              ('Golgi-Nucleus cluster','#00A0A0'), # Golgi nuc
                     'CHLOROPLAST':      ('Chloroplast','#50FF20'),
	             'CHROMATIN':        ('Nuclear chromatin','#905020'),
	             'NUCLEUS-CHROMATIN':('Nuclear chromatin','#905020'),
                     'CNR':              ('Ribosome-adjascent','#A0FFA0'), # Next to ribo
                     'CYTOSOL':          ('Cytosol','#FF0000'),
                     'DUAL':             ('Dual localised','#DDDDDD'),
                     'ENDOSOME':         ('Endosome','#B080FF'),
                     'ER':               ('Endoplasmic reticulum','#0050FF'),
                     'ECM':              ('Exctracellular','#FFFFFF'),
                     'EXTRACELLULAR':    ('Exctracellular','#FFFFFF'),
                     'GOLGI CIS':        ('cis-Golgi','#00B0B0'),
                     'GOLGI':            ('Golgi apparatus','#FFD000'),
                     'GOLGI MEDIAL':     ('Medial Golgi','#FFE000'),
                     'GOLGI TRANS':      ('trans-Golgi','#FF4000'),
                     'LYSOSOME':         ('Lysosome','#009000'),
                     'MITOCHONDRION':    ('Mitochontrion','#FF7000'),
                     'MIM':              ('Mitochondrial inner membrane','#FF4000'),
                     'MOM':              ('Mitochondrial outer membrane','#0080FF'),
                     'NUCLEUS':          ('Nucleus','#80B0F0'),
                     'PEROXISOME':       ('Peroxisome','#FF80FF'),
                     'PLASMODESMATA':    ('Plasmodesma','#FFFF80'),
                     'PLASTID MEMBRANE': ('Plastid outer membrane','#0080FF'),
                     'PLASTID STROMA':   ('Plastid stroma','#DDDD00'),
                     'PM':               ('Plasma membrane','#FF00FF'),
                     'PMS':              ('PM sub-cluster','#40FFFF'), # PM sub pop
                     'RIBOSOMAL':        ('Ribosome','#7020FF'),
                     'TGN':              ('trans Golgi network','#B080FF'),
	             'UNKNOWN':          ('Unkown/unclassified','#404040'),
                     'VACUOLE':          ('Vacuole','#00A000'),
                     'RIBOSOME 40S':     ('Ribosome 40S subunit', '#000080'),
                     'RIBOSOME 60S':     ('Ribosome 60S subunit', '#800000'),
                     'ER/GOLGI':         ('ER and Golgi apparatus', '#CCCC00'),
                     'CYTOSKELETON':     ('Actin cytoskeleton', '#008080'),
                     'PROTEASOME':       ('Proteasome complex', '#40D000'),
                     'G1':               ('Test Group 1', '#00A0A0'),
                     'G2':               ('Test Group 2', '#FFEEDD'),
                     'G3':               ('Test Group 3', '#A0FFA0'),
            }
            
  
class AnsiColors(object):

    _ansi_esc_color_dict = {"end":'\033[0m',
                            "bold":'\033[1m',
                            "italic":'\033[2m',
                            "underline":'\033[3m',
                            "black":'\033[30m',
                            "red":'\033[31m',
                            "green":'\033[32m',
                            "yellow":'\033[33m',
                            "blue":'\033[34m',
                            "magenta":'\033[35m',
                            "cyan":'\033[36m',
                            "white":'\033[37m',
                            "grey":'\033[90m',
                            "lt_red":'\033[91m',
                            "lt_green":'\033[92m',
                            "lt_yellow":'\033[93m',
                            "lt_blue":'\033[94m',
                            "lt_magenta":'\033[95m',
                            "lt_cyan":'\033[96m',
                            "lt_white":'\033[97m',}
    
    def __init__(self):
       for key, val in self._ansi_esc_color_dict.items():
           setattr(self, key, val)   

    def wrap(self, text, color='blue'):
       
       return f"{self._ansi_esc_color_dict[color]}{text}{self._ansi_esc_color_dict['end']}"
       
