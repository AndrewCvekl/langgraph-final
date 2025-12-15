"""Catalog tools for browsing the music store.

All tools are read-only and return structured data including
TrackId and UnitPrice where applicable.
"""

from langchain_core.tools import tool

from src.db import get_db


@tool
def list_genres() -> str:
    """List all available music genres in the store.
    
    Returns:
        A list of all genre names.
    """
    db = get_db()
    result = db.run(
        "SELECT GenreId, Name FROM Genre ORDER BY Name;",
        include_columns=True
    )
    return result


@tool
def artists_in_genre(genre_name: str) -> str:
    """Get all artists in a specific genre.
    
    Args:
        genre_name: The name of the genre (e.g., "Rock", "Jazz").
        
    Returns:
        A list of artists that have tracks in the specified genre.
    """
    db = get_db()
    result = db.run(
        f"""
        SELECT DISTINCT Artist.ArtistId, Artist.Name as ArtistName
        FROM Artist
        JOIN Album ON Artist.ArtistId = Album.ArtistId
        JOIN Track ON Album.AlbumId = Track.AlbumId
        JOIN Genre ON Track.GenreId = Genre.GenreId
        WHERE Genre.Name LIKE '%{genre_name}%'
        ORDER BY Artist.Name
        LIMIT 50;
        """,
        include_columns=True
    )
    return result


@tool
def albums_by_artist(artist_name: str) -> str:
    """Get all albums by a specific artist.
    
    Args:
        artist_name: The name of the artist (partial match supported).
        
    Returns:
        A list of albums by the artist, including album ID and title.
    """
    db = get_db()
    result = db.run(
        f"""
        SELECT Album.AlbumId, Album.Title as AlbumTitle, Artist.Name as ArtistName
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE '%{artist_name}%'
        ORDER BY Album.Title;
        """,
        include_columns=True
    )
    return result


@tool
def tracks_in_album(album_title: str) -> str:
    """Get all tracks in a specific album.
    
    Args:
        album_title: The title of the album (partial match supported).
        
    Returns:
        A list of tracks with TrackId, name, duration, and price.
    """
    db = get_db()
    result = db.run(
        f"""
        SELECT 
            Track.TrackId,
            Track.Name as TrackName,
            Album.Title as AlbumTitle,
            Artist.Name as ArtistName,
            Track.Milliseconds / 1000 as DurationSeconds,
            Track.UnitPrice
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Album.Title LIKE '%{album_title}%'
        ORDER BY Track.TrackId;
        """,
        include_columns=True
    )
    return result


@tool
def find_track(track_query: str) -> str:
    """Search for tracks by name or TrackId.
    
    Args:
        track_query: The track name to search for, or a TrackId number.
        
    Returns:
        Matching tracks with TrackId, name, artist, album, and price.
    """
    db = get_db()
    
    # Check if query is a numeric TrackId
    try:
        track_id = int(track_query)
        result = db.run(
            f"""
            SELECT 
                Track.TrackId,
                Track.Name as TrackName,
                Album.Title as AlbumTitle,
                Artist.Name as ArtistName,
                Genre.Name as GenreName,
                Track.Milliseconds / 1000 as DurationSeconds,
                Track.UnitPrice
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            WHERE Track.TrackId = {track_id};
            """,
            include_columns=True
        )
    except ValueError:
        # Search by name
        result = db.run(
            f"""
            SELECT 
                Track.TrackId,
                Track.Name as TrackName,
                Album.Title as AlbumTitle,
                Artist.Name as ArtistName,
                Genre.Name as GenreName,
                Track.Milliseconds / 1000 as DurationSeconds,
                Track.UnitPrice
            FROM Track
            JOIN Album ON Track.AlbumId = Album.AlbumId
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Genre ON Track.GenreId = Genre.GenreId
            WHERE Track.Name LIKE '%{track_query}%'
            ORDER BY Track.Name
            LIMIT 20;
            """,
            include_columns=True
        )
    
    return result


