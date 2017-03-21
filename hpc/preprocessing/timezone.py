import datetime
import numpy as np

state2tz= {
 'AK': -9,
 'AL': -6,
 'AR': -6,
 'AZ': -7,
 'CA': -8,
 'CO': -7,
 'CT': -5,
 'DC': -5,
 'DE': -5,
 'FL': -5,
 'GA': -5,
 'HI': -10,
 'IA': -6,
 'ID': -7,
 'IL': -6,
 'IN': -5,
 'KS': -6,
 'KY': -5,
 'LA': -6,
 'MA': -5,
 'MD': -5,
 'ME': -5,
 'MI': -5,
 'MN': -6,
 'MO': -6,
 'MS': -6,
 'MT': -7,
 'NC': -5,
 'ND': -6,
 'NE': -6,
 'NH': -5,
 'NJ': -5,
 'NM': -7,
 'NV': -8,
 'NY': -5,
 'OH': -5,
 'OK': -6,
 'OR': -8,
 'PA': -5,
 'RI': -5,
 'SC': -5,
 'SD': -6,
 'TN': -6,
 'TX': -6,
 'UT': -7,
 'VA': -5,
 'VT': -5,
 'WA': -8,
 'WI': -6,
 'WV': -5,
 'WY': -7}

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

states_inverted = {v:k for (k,v) in states.items()}


def map2name(s):
    if "USA" in s:
        state = s[:-5]
        if state in states_inverted:
            return states_inverted[state]
        else:
            return "<Other>"
    try:
        state_code = s[-2:]
        if state_code in states:
            return state_code
    except:
        return "<Other>"
    return "<Other>"


def map2tz(s):
    try:
        return state2tz[s]
    except:
        return 0


def convert2local(df):
    # df: index is datetime, has "place" column
    # new years dataset

    # convert the place name into a state or <other>
    df["location"] = df.place.astype(str).apply(map2name)

    # convert the state to a timedelta object
    df["tz"] = df.location.apply(map2tz).apply(lambda h: datetime.timedelta(hours=np.asscalar(h)))

    # modify the index to account for the timedelta
    df['local_time'] = df.index + df["tz"]

    # filter data only that have state data
    return df[df["location"] != "<Other>"]