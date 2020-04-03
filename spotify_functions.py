'''
Katie Connor, Serena Luo, Anna Repp
DS 2000 Final Project Code
April 3rd, 2020
'''

import os
import sys
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import math
import pandas as pd
from pandas.core.common import flatten
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud

# Set environmental variables for authentication
# To get client_id and client_secret, apply to use the Spotify API at:
# https://developer.spotify.com/
os.environ['SPOTIPY_CLIENT_ID'] = # INSERT CLIENT ID HERE
os.environ['SPOTIPY_CLIENT_SECRET'] = # INSERT CLIENT SECRET HERE
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://google.com/'

# Set function variables
TOP_GENRES = 30
ATTR_LIST = ['danceability', 'energy', 'acousticness',
             'instrumentalness', 'speechiness', 'valence']

# DEFINE PLAYLIST CLASS --------------------------------------------------------
class Playlist(object):

        def __init__(self, username, p_id, length, tracks):
            ''' Class: Playlist
                Stores: Attributes and methods to describe a single playlist
                Parameters: username (str), p_id (str) representing the Spotify
                            URI for a playlist, length of playlist (int),
                            tracks (pandas DataFrame)'''
            self.username = username
            self.p_id = p_id
            self.length = length
            self.tracks = tracks

        def avg_attr(self, attr):
            ''' Method: avg_attr
                Parameters: attribute (str)
                Returns: average value of given attribute in playlist tracks
            '''
            avg = self.tracks[attr].mean()
            return avg

        def all_artists(self):
            ''' Method: all_artists
                Parameters: none
                Returns: a list of all artists in tracks['artists']
                         NOTE: MAY CONTAIN DUPLICATES
            '''
            artistlist = []
            artistcol = self.tracks['artist_ids']
            for i in range(len(artistcol)):
                ids = artistcol[i]
                for id in ids:
                    artistlist.append(id)
            return artistlist

        def all_genres(self):
            ''' Method: all_genres
                Parameters: none
                Returns: a list of all genres in tracks['genres']
                         NOTE: MAY CONTAIN DUPLICATES
            '''
            genrelist = []
            genrecol = self.tracks['artist_genres']
            for i in range(len(genrecol)):
                genres = genrecol[i]
                for genre in genres:
                    genrelist.append(genre)
            return genrelist

# GET USER DATA
def get_playlist_uri(username):
    ''' Function: get_playlist_uri
        Parameters: username (string)
        Returns: the unique identification code of a user's 2019 Wrapped
                 playlist (as string)
    '''
    # Create Spotify object and pull user's public playlists
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    playlists = sp.user_playlists(username)

    # Check user's public playlists for Wrapped 2019 playlist
    top_2019_songs = 'Your Top Songs 2019'

    while playlists:
        # Break loop once playlist is found
        for i, playlist in enumerate(playlists['items']):
            if playlist['name'] == top_2019_songs:
                uri = playlist['uri']
                break
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None

    # Error message if playlist does not exist/is not public
    if uri is None:
        print('Could not find playlist, please make sure "Your Top Songs 2019" '
              'is set to public')

    return uri

def get_playlist_tracks(playlist_id, user):
    ''' Function: get_playlist_tracks
        Parameters: playlist_id (str), user name (string)
        Returns: a Playlist object containing playlist and track data
    '''
    print('Getting Wrapped 2019 tracks...')
    # Create Spotify object to interact with API
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    # Get playlist tracks
    tracks = sp.playlist_tracks(playlist_id)
    items = tracks['items']

    # Iterate through each track id, getting detailed information for song,
    # artist, genre, and audio features
    trackinfo = []
    for item in items:
        # Track info
        trackid = item['track']['id']
        trackname = item['track']['name']

        # Artist / genre info
        artists = []
        artistids = []
        artistgenres_all = []
        for songartist in item['track']['album']['artists']:
            artistname = songartist['name']
            artistid = songartist['id']
            artists.append(artistname)
            artistids.append(artistid)
            artistinfo = sp.artist(artistid)
            artistgenres = artistinfo['genres']
            for genre in artistgenres:
                artistgenres_all.append(genre)

        # Audio feature info
        aud_feat = sp.audio_features(trackid)[0]
        dance = aud_feat['danceability']
        energy = aud_feat['energy']
        acoustic = aud_feat['acousticness']
        instrument = aud_feat['instrumentalness']
        speech = aud_feat['speechiness']
        valence = aud_feat['valence']

        # Put all track data in a list, append to master list
        info = [trackid, trackname, artists, artistids, artistgenres_all,
                dance, energy, acoustic, instrument, speech, valence]
        trackinfo.append(info)

    # Convert master list to DataFrame
    trackinfo = pd.DataFrame(trackinfo)
    trackinfo.columns = ['track_id', 'track_name', 'artists', 'artist_ids',
                         'artist_genres', 'danceability', 'energy',
                         'acousticness', 'instrumentalness', 'speechiness',
                         'valence']

    # Put DataFrame and other playlist data into a Playlist object
    plist = Playlist(username = user, p_id = playlist_id,
                     length = len(trackinfo), tracks = trackinfo)

    return plist

def get_user_data(username):
    ''' Function: get_user_data
        Parameters: username (str)
        Returns: a Playlist object for their 2019 Wrapped playlist.
                 (This function just wraps get_playlist_uri and
                  get_playlist_tracks together)
    '''
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    uri = get_playlist_uri(username)
    data = get_playlist_tracks(uri, username)

    return data

# USER AUTHORIZATION / PULLING FROM API ----------------------------------------
def authorize_main_user(username):
    ''' Function: authorize_main_user
        Parameters: Spotify username (str)
        Returns: access (str) determining if account was successfully
                 authorized w/API
    '''
    # Get authorization token
    scope = 'playlist-modify-public'
    token = util.prompt_for_user_token(username, scope)

    # Determines access status
    if token:
        access = 'Approved'
    else:
        print("Authorization Failed")
        access = 'Failed'

    return access

def get_playlists():
    ''' Function: get_playlists
        Parameters: none
        Returns: a list of Playlist objects for an inputted group of users
    '''
    # Initialize empty list to store Playlists
    master_list = []

    # Authorize main user
    main_user = input('Please enter your username: ')
    access = authorize_main_user(main_user)

    # Check access. If denied, end program
    if access != 'Approved':
        sys.exit()

    # Get playlist info for main user
    master_list.append(get_user_data(main_user))

    # Add playlist info for second user (required)
    user2 = input('Please enter another username for comparison: ')
    master_list.append(get_user_data(user2))

    # Prompt for more usernames to add (optional)
    end = False
    while end == False:
        cont = int(input('Enter 0 to add another user. To move on, enter 1: '))
        if cont == 1:
            break
        elif cont == 0:
            username = input('Please enter another username for comparison: ')
            master_list.append(get_user_data(username))
        elif cont != 0 and cont != 1:
            print('Invalid code. Please enter 0 or 1.')

    return master_list

# PLAYLIST/CSV CONVERSIONS -----------------------------------------------------
def playlists_to_csv(playlists, filename):
    ''' Function: playlists_to_csv
        Parameters: list of Playlist objects, filename to save to (str)
        Returns: nothing
        Does: converts the Playlists to a DataFrame, then saves DF as CSV
    '''
    # Initialize list to store DataFrames
    frames = []

    # Iterate through list of Playlists, converting each to a DataFrame by
    # adding a column for playlist-level info such as username, p_id, and length
    for playlist in playlists:
        converted = playlist.tracks
        converted['p_id'] = playlist.p_id
        converted['length'] = playlist.length
        converted['user'] = playlist.username
        frames.append(converted)

    # Concatenate the DataFrames together and write to CSV
    df = pd.concat(frames, sort = False)
    df.to_csv(filename, index=False)

def relist(string):
    ''' Function: relist
        Parameters: string
        Returns: string values in a list. Used to re-import data from CSV with
                 list columns.
    '''
    converted = string.replace("'", '')
    converted = converted.strip("][").split(', ')
    return converted

def csv_to_playlists(filename):
    ''' Function: csv_to_playlists
        Parameters: file name (str)
        Returns: a list of Playlist objects
    '''
    # Read CSV
    csvdata = pd.read_csv(filename)

    # Initialize empty list to store Playlists
    playlists = []

    # Get all unique users in CSV
    userlist = np.unique(csvdata['user'])
    userlist = userlist.tolist()

    # Iterate through users, isolating username/p_id/length from CSV data and
    # converting track data to a dataframe
    for user in userlist:
        userdata = csvdata.loc[csvdata['user'] == user]
        user_p_id = userdata.iloc[0]['p_id']
        user_length = userdata.iloc[0]['length']
        userdata = userdata.drop('p_id', 1)
        userdata = userdata.drop('length', 1)
        userdata = userdata.drop('user', 1)

        # Convert string lists to real lists
        userdata['artists'] = userdata['artists'].apply(relist)
        userdata['artist_ids'] = userdata['artist_ids'].apply(relist)
        userdata['artist_genres'] = userdata['artist_genres'].apply(relist)

        userdata = userdata.reset_index(drop = True)

        # Create Playlist object for user
        p_obj = Playlist(username = user, p_id = user_p_id,
                         length = user_length, tracks = userdata)

        playlists.append(p_obj)

    return playlists

# SIMILARITY CALCULATION -------------------------------------------------------

def songs_diffpct(playlist1, playlist2):
    ''' Function: songs_diffpct
        Parameters: two Playlist objects
        Returns: percent (float) of songs (out of all songs from both playlists)
                 that were NOT shared
    '''
    # Get all track IDs for each playlist
    songs1 = playlist1.tracks['track_id'].values.tolist()
    songs2 = playlist2.tracks['track_id'].values.tolist()

    # Get total number of songs in both playlists
    totalsongs_withdupes = len(songs1) + len(songs2)

    # Count number of songs common to both playlists
    countshared = 0
    for song in songs1:
        if song in songs2:
            countshared += 1

    # Adjust total song number to account for duplicate track_ids from shared
    # songs
    distinct_songs = totalsongs_withdupes - countshared
    # Get different songs by subtracting the number of shared songs from
    # adjusted total
    diff_songs = totalsongs_withdupes - (countshared * 2)

    # Calculate % songs that are different between users
    diff_pct = diff_songs / distinct_songs

    return diff_pct

def artists_diffpct(playlist1, playlist2):
    ''' Function: artists_diffpct
        Parameters: two Playlist objects
        Returns: percent (float) of artists (out of all artists from both
                 playlists) that were NOT shared
    '''
    # Get all artist IDs for each playlist
    artists1 = set(playlist1.all_artists())
    artists2 = set(playlist2.all_artists())

    # Get total number of artists in both playlists
    totalartists_withdupes = len(artists1) + len(artists2)

    # Count number of artists common to both playlists
    countshared = 0
    for artist in artists1:
        if artist in artists2:
            countshared += 1

    # Adjust total artist number to account for duplicate artist_ids from shared
    # artists
    distinct_artists = totalartists_withdupes - countshared
    # Get different artists by subtracting the number of shared artists from
    # adjusted total
    diff_artists = totalartists_withdupes - (countshared * 2)

    # Calculate % artists that are different between users
    diff_pct = diff_artists / distinct_artists

    return diff_pct

def count_genres(playlist, topn):
    ''' Function: count_genres
        Parameters: Playlist object, top n (int) genres to return
        Returns: Most common genres appearing in one playlist
    '''
    # Get all genres from playlist
    genres = playlist.all_genres()
    # Store all distinct genres in separate list
    distinct_genres = set(genres)

    # Initialize list to store appearance counts for each genre
    genre_data = []
    # For each genre, count appearances in original genre list and append count
    # to storage list
    for genre in distinct_genres:
        genre_count = genres.count(genre)
        genre_data.append([genre, genre_count])

    # Convert storage list to DF, sort by count (high to low), select top n rows
    genre_data = pd.DataFrame(genre_data, columns = ['genre', 'count'])
    genre_data = genre_data.sort_values(by='count', ascending = False)
    genre_data = genre_data.reset_index(drop = True)
    top_genres = genre_data.iloc[0:(topn+1), 0]

    # Return top n genres
    return top_genres

def genres_diffpct(playlist1, playlist2):
    ''' Function: genres_diffpct
        Parameters: two Playlist objects
        Returns: percent (float) of genres (out of all genres from both
                 playlists) that were NOT shared
    '''
    # Get top genres from both playlists
    genres1 = count_genres(playlist1, TOP_GENRES)
    genres2 = count_genres(playlist2, TOP_GENRES)

    # Count shared genres between playlists
    merge = pd.merge(genres1, genres2, on='genre', how='inner')
    common = len(merge)

    # Calculate distinct genres, non-shared genres, and % of non-shared genres
    distinct_genres = (TOP_GENRES * 2) - common
    diff_genres = (TOP_GENRES * 2) - (common * 2)
    diff_pct = diff_genres / distinct_genres

    return diff_pct

def eucl_dist(p1, p2):
    ''' Function: eucl_dist
        Parameters: p1, p2 (both Playlist objects)
        Returns: euclidean distance between two playlists, based on distance of
                 average audio feature and song/artist/genre similarity
    '''
    # Calculate difference between p1 and p2 average audio features
    dance = p1.avg_attr('danceability') - p2.avg_attr('danceability')
    energy = p1.avg_attr('energy') - p2.avg_attr('energy')
    acoust = p1.avg_attr('acousticness') - p2.avg_attr('acousticness')
    instru = p1.avg_attr('instrumentalness') - p2.avg_attr('instrumentalness')
    speech = p1.avg_attr('speechiness') - p2.avg_attr('speechiness')
    valence = p1.avg_attr('valence') - p2.avg_attr('valence')

    # Get difference %s for artist, song, genre
    artist = artists_diffpct(p1, p2)
    songs = songs_diffpct(p1, p2)
    genre = genres_diffpct(p1, p2)

    # Calculate sum of squares for the differences
    sum_squares = (dance**2 + energy**2 + acoust**2 + instru**2 + speech**2
                + valence**2 + artist**2 + songs**2 + genre**2)

    # Take square root to get eucl dist
    eucl_dist = math.sqrt(sum_squares)

    return eucl_dist

# GROUP PLAYLIST GENERATOR -----------------------------------------------------
def get_feature_range(playlist_list, feature):
    ''' Function: get_feature_range
        Parameters: list of Playlist objects, feature (str)
        Returns: list containing feature name, min/max/average of the average
                 audio feature scores in the group
    '''
    # Initialize list to store all average feature scores
    user_avgs = []
    # Iterate through list of playlists, appending average feature score to list
    for playlist in playlist_list:
        avg = playlist.avg_attr(feature)
        user_avgs.append(avg)

    # Get min/max/avg
    feat_min = min(user_avgs)
    feat_max = max(user_avgs)
    feat_avg = np.mean(user_avgs)

    # Return list of results
    return [feature, feat_min, feat_max, feat_avg]

def all_feature_ranges(playlist_list):
    ''' Function: all_feature_ranges
        Parameters: list of Playlists
        Returns: dict of each audio feature and its min/max/avg
    '''
    features = ATTR_LIST

    # Initialize empty dict
    feat_dict = {}
    # Iterate through list of features, getting stats and appending them to dict
    # with feature as key
    for feature in features:
        vals = get_feature_range(playlist_list, feature)
        feat_dict[feature] = vals[1:]

    return feat_dict

def group_genres(playlist_list):
    ''' Function: group_genres
        Parameters: list of Playlists
        Returns: list of all genres that appear in more than one Playlist
    '''
    # Init list to store all genre info
    all_genres = []
    # Iterate through list of Playlists, appending its top n genres to list
    for playlist in playlist_list:
        genres = count_genres(playlist, TOP_GENRES)
        genres = genres.values.tolist()
        # Need to iterate through genres otherwise the whole list will be
        # appended and genres can't be analyzed later
        for genre in genres:
            all_genres.append(genre)

    # Make a list of distinct genres
    distinct_genres = set(all_genres)

    # Initialize list to store genre count data
    genre_data = []
    # For each distinct genre, count # appearances in all_genres. Append count.
    for genre in distinct_genres:
        genre_count = all_genres.count(genre)
        genre_data.append([genre, genre_count])

    # Convert to dataframe and select all genres where count > 1
    genre_data = pd.DataFrame(genre_data, columns = ['genre', 'count'])
    genre_data = genre_data.sort_values(by='count', ascending = False)
    shared = genre_data.loc[genre_data['count'] > 1]
    # Convert resulting DF back to list
    shared = shared['genre'].values.tolist()

    return(shared)

def in_genres(track_genres, genre_list):
    ''' Function: in_genres
        Parameters: track_genres (list of strings), genre_list (list of strings)
        Returns: Bool for whether or not a genre in track_genres appears in
                 genre_list
    '''
    genre_match = False

    # Iterate through track genres. If any genre appears in genre_list, break
    # loop and return updated bool
    for i in range(len(track_genres)):
        genre = track_genres[i]
        if genre in genre_list:
            genre_match = True
            break

    return genre_match

def select_common_tastes(playlist_list):
    ''' Function: select_common_tastes
        Parameters: list of Playlists
        Returns: DataFrame of tracks that generally align with common genre/
                 audio feature tastes of all input playlists
    '''
    # Get group common genres
    genres = group_genres(playlist_list)
    # Get group audio feature stats (min/max/avg)
    ranges = all_feature_ranges(playlist_list)

    # Create a DF for all tracks in all playlists
    alltracks = []
    for playlist in playlist_list:
        tracks = playlist.tracks.copy()
        alltracks.append(tracks)
    data = pd.concat(alltracks, sort=False)

    # Create new column showing if track is in common genre list
    data['valid_genre'] = data['artist_genres'].apply(in_genres, args=(genres,))

    # Create filtered DataFrame where audio features in track must be between
    # defined min/max values
    data_filtered = data[(data['valid_genre'] == True) &
                         (data['danceability'] >= ranges['danceability'][0]) &
                         (data['danceability'] <= ranges['danceability'][1])]# &
                               # (data['energy'] >= ranges['energy'][0]) &
                               # (data['energy'] <= ranges['energy'][1]) &
                               # (data['valence'] >= ranges['valence'][0]) &
                               # (data['valence'] <= ranges['valence'][1])]

    # Select key columns to return, remove duplicate track rows
    return_tracks = data_filtered[['track_id', 'track_name', 'artists',
                                   'artist_ids']]
    return return_tracks

# SINGLE USER MUSIC RECOMMENDATION ---------------------------------------------
# NOTE: CODE CURRENTLY NOT IN USE. THIS WOULD BE THE START OF A DIFFERENT SONG
# RECOMMENDATION PROCESS.
def rank_similar_listeners(playlist_list, user):
    ''' Function: rank_similar_listeners
        Parameters: list of Playlists, user name (str) - must be an existing
                    playlist.user
        Returns: sorted DataFrame of all other playlist.users, by similarity
    '''
    # Make list of users to compare to
    otherusers = []
    for playlist in playlist_list:
        # Define main playlist
        if playlist.username == user:
            mainplaylist = playlist
        # Append all other playlists to storage list
        else:
            otherusers.append(playlist)

    # Initialize list to store users and scores
    userscores = []
    # For each non-main user playlist, calculate similarity and append result
    for playlist in otherusers:
        user = playlist.username
        score = eucl_dist(mainplaylist, playlist)
        userscores.append([user, round(score,2)])

    # Convert scores list to DF and reformat
    userscores = pd.DataFrame(userscores)
    userscores.columns = ['user', 'similarityscore']
    userscores = userscores.sort_values(by = 'similarityscore')
    userscores = userscores.reset_index(drop = True)

    return userscores

# VISUALIZATIONS AND RELATED FUNCTIONS -----------------------------------------
def viz_features(playlists):
    ''' Function: viz_features
        Parameters: list of Playlists
        Returns: none
        Does: Displays chart of average audio feature scores for every playlist
              in list, across six audio features
    '''
    # Create list of lists of average audio feature scores
    data = []
    for playlist in playlists:
        playlistinfo = []
        for attribute in ATTR_LIST:
            attr = playlist.avg_attr(attribute)
            playlistinfo.append(attr)
        data.append(playlistinfo)
    # Convert to DataFrame
    data = pd.DataFrame(data)
    data.columns = ATTR_LIST

    # Create a Y series to plot (0 because no need for 2nd dimension)
    y = [0] * len(data)

    # Create a plot with 6 subplots
    fig, axs = plt.subplots(6,1)

    # Plot each audio feature as a scatter
    for i in range(0, 6):
        axs[i].scatter(data.iloc[:,i], y, color = 'lightgreen')
        axs[i].set_xlim(0,1)
        axs[i].yaxis.set_ticklabels([])
        axs[i].set_ylabel(data.columns[i], rotation = 0,
                          fontsize = 12, ha = 'right', va = 'center')
        # Remove ticks for all but one plot (reduces extra info)
        if i != 5:
            axs[i].xaxis.set_ticklabels([])

    #Title plot
    fig.suptitle('Average Audio Feature Scores (All Group Members)', size = 16)

    # Display plot
    plt.show

def pair_scores(playlist_list):
    ''' Function: pair_scores
        Parameters: list of Playlists
        Returns: list containing every possible pair of playlists in list and
                 the pair's similarity score
    '''
    # Initialize list to store score info for all pairs
    scores = []

    # Iterate through list, taking each playlist and calculating similarity
    # scores for all following playlists
    for i in range(len(playlist_list)-1):
        p1 = playlist_list[i]
        otherp = playlist_list[i+1:]
        name1 = p1.username
        for j in range(len(otherp)):
            p2 = otherp[j]
            score = eucl_dist(p1, p2)
            name2 = p2.username
            # Append both playlist usernames and their similarity score to list
            scores.append([name1, name2, score])

    return scores

def get_range(scorelist):
    ''' Function: get_range
        Parameters: list of lists (sublists contain two usernames and their
                    similarity score)
        Returns: minimum and maximum similarity score
    '''
    # Pull all similarity scores from input list
    values = []
    for row in scorelist:
        score = row[2]
        values.append(score)

    # Get min and max scores
    minscore = min(values)
    maxscore = max(values)

    # Return min, max as tuple
    return (minscore, maxscore)

def spread_scores(scorelist):
    ''' Function: spread_scores
        Parameters: list of lists (sublists contain two usernames and their
                    similarity score)
        Returns: scorelist with additional field per list representing where the
                 original score lies on a scale of 1-10 compared to the min and
                 max scores. This is to make understanding which user pairs are
                 more/less similar easier.
    '''
    # Get min, max, and range of scores in scorelist
    scorevals = get_range(scorelist)
    minscore = scorevals[0]
    maxscore = scorevals[1]
    scorerange = maxscore - minscore

    # Calculate "spread" score
    for row in scorelist:
        score = row[2]
        newscore = ((score - minscore) / scorerange) * 10
        row.append(newscore)

    return scorelist

def plot_network(playlist_list):
    ''' Function: plot_network
        Parameters: list of playlists
        Returns: none
        Does: Plots a network of all playlists, labeling each edge with the
              similarity scores for the node pair
    '''
    # Create a graph object
    G = nx.Graph()
    # Get all similarity scores (original and spread) for playlist list
    scores = spread_scores(pair_scores(playlist_list))

    # Draw graph edges
    edgelabels = {}
    # For each pair of playlists, add an edge between those nodes and append
    # score to a list of edge labels to be applied later
    for row in spread_scores(scores):
        node1 = row[0]
        node2 = row[1]
        score = round(row[3],2)
        G.add_edge(node1, node2, length = score) # LENGTH ARGUMENT NOT WORKING
        edgelabels[(node1, node2)] = score

    # Draw graph with circular layout (no nodes in center)
    pos = nx.circular_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color = 'lightgreen')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgelabels)
    nx.draw_networkx_labels(G, pos, with_labels=True)

    # Add titles
    plt.title('Similarity Scores between Users', fontsize = 20)
    plt.suptitle('0 = highest similarity in group, '
                 '10 = lowest similarity in group', fontsize = 10)

    plt.show()

def genre_words(playlist_list):
    ''' Function: genre_words
        Parameters: list of Playlists
        Returns: string of all genres occuring in playlists, with spaces and
                 punctuation removed (to generate wordcloud later)
    '''
    # Make list of all genres in playlists
    allgenres = []
    for playlist in playlist_list:
        data = count_genres(playlist, 30).values.tolist()
        for value in data:
            allgenres.append(value)

    # Replace spaces and hyphens in each genre
    genredata = []
    for genre in allgenres:
        genre = genre.replace(' ', '')
        genre = genre.replace('-', '')
        genredata.append(genre)

    # Turn list of all genres into one string
    genrestring = ' '.join(genredata)

    return genrestring

def generate_wordcloud(data):
    ''' Function: generate_wordcloud
        Parameters: string
        Returns: none
        Does: plots a wordcloud from input string
    '''
    wordcloud = WordCloud(background_color='white',
                          max_words=100).generate(data)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

def all_viz(playlist_list):
    ''' Function: all_viz
        Parameters: list of playlists
        Returns: none. Creates multiple plots (to reduce user effort).
                 CURRENTLY NOT WORKING BECAUSE PLOTS GET SQUISHED
    '''
    viz_features(playlist_list)
    plot_network(playlist_list)
    generate_wordcloud(genre_words(playlist_list))

# WRITING RECOMMENDED PLAYLISTS ------------------------------------------------

def create_playlist(main_user, playlist_name, token):
    ''' Function: create_playlist
        Parameters: username (str), playlist name (str), token (str)
        Returns: none
        Does: creates a new playlist in the authorized Spotify account
    '''
    # Make Spotify object, create playlist, return playlist URI
    sp = spotipy.Spotify(auth = token)
    playlists = sp.user_playlist_create(main_user, playlist_name)
    uri = playlists['uri']
    return uri

def authorize_main_user_token(username):
    ''' Function: authorize_main_user_token
        Parameters: account username (string)
        Returns: access token to use as authorization with API
    '''
    scope = 'playlist-modify-public'
    token = util.prompt_for_user_token(username, scope)
    return token

def add_tracks(main_user, uri, track_ids, token):
    ''' Function: add_tracks
        Parameters: username (str), playlist URI (str), list of track IDs (list
                    of str), authorization token (str)
        Returns: none
        Does: adds all tracks in list of track_ids to given Spotify playlist
    '''
    # Create Spotify object with authorization
    sp = spotipy.Spotify(auth=token)
    results = sp.user_playlist_add_tracks(main_user, uri, track_ids)

def create_spotify_playlist(songrecs):
    ''' Function: create_spotify_playlist
        Parameters: DataFrame containing list of song recommendations (columns =
                    ['track_id', 'track_name', 'artists', 'artist_ids'])
        Returns: none
        Does: prompts for user to input their username and a playlist name, then
              creates a playlist in Spotify with all of the parameter songs
    '''
    # Get username and playlist name
    main_user = input('Please enter your Spotify username: ')
    playlist_name = input('Please enter a playlist name: ')

    # Authorize user and create a playlist
    token = authorize_main_user_token(main_user)
    uri = create_playlist(main_user, playlist_name, token)

    # Get track IDs from input DataFrame
    songrecs = songrecs['track_id'].values.tolist()

    # Add tracks to playlist in batches of 100 (API limit)
    while len(songrecs) > 0:
        if len(songrecs) > 100:
            addtracks = songrecs[0:100]
            songrecs = [x for x in songrecs if x not in addtracks]
            add_tracks(main_user, uri, addtracks, token)
        elif len(songrecs) > 0:
            addtracks = songrecs
            songrecs = []
            add_tracks(main_user, uri, addtracks, token)

    # Notify user when complete
    print('All set! Please check your Spotify account for the new playlist.')
