SELECT t.track_id, t.title, t.artists, t.album, t.popularity, 
f.danceability, f.energy, f.key, f.loudness, f.mode, f.speechiness, f.acousticness, f.instrumentalness, f.liveness, f.valence, f.tempo, f.duration_ms, f.time_signature,
p.genre
FROM FEATURES f
JOIN TRACKS t on t.track_id = f.track_id
JOIN PLAYLISTS p on p.playlist_id = t.playlist_id
