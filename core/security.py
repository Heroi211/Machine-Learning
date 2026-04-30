"""Provide password hashing and verification helpers."""

from passlib.context import CryptContext

CRYPT = CryptContext(schemes=["bcrypt"],deprecated="auto")

def verify_password(password:str,hash_password:str) ->bool:
    """Return whether a plaintext password matches a stored hash."""
    return CRYPT.verify(password,hash_password)

def get_password_hash(password:str)->str:
    """Return a secure hash for a plaintext password."""
    return CRYPT.hash(password)
