#! /usr/bin/python3

import seaborn as sns
#########################################
# colors
#########################################
RED_1 = "#C55659"
RED_2 = "#D22027"
BLACK_1 = '#171717'

BLUE_1 = '#7FA5B7'
BLUE_2 = '#385989'
BLUE_3 = '#35547C'

CYAN_1 = '#5BB7CD'

ORANGE_1 = '#F7903D'

YELLOW_1 = '#CBB47B'
YELLOW_2 = '#F7CE7D'

GREEN_1 = '#68BA9F'
GREEN_2 = '#54AC75'

PURPLE_1 = '#7572B5'

PINK_1 = '#DE5D78'

RED_2 = "#D22027"
BLUE_2 = '#385989'
BLACK_1 = '#171717'

PALETTE = {
    1:
    {
        'red': RED_2,
        'blue': BLUE_2
    },
    2:
    {
        'eq': [RED_2, BLUE_2],
        'diff': [RED_2, BLACK_1]
    },
    3:
    {
        'eq': [BLUE_2, ORANGE_1, GREEN_2],
        'diff': [RED_2, BLUE_2, BLUE_1]
    },
    4:
    {
        'eq': [BLUE_2, GREEN_1, PINK_1, YELLOW_2]
    },
    5:
    {
        'eq': [YELLOW_1, CYAN_1, GREEN_2, RED_1, PURPLE_1]
    },
    10: {
        'eq' : sns.color_palette('muted')
    },
    16: {
        'eq' : sns.color_palette("husl", 16)
    }
}

