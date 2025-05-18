import hashlib
import base64
import numpy as np

class StreamEncryption:
    def __init__(self, key=None):
        # Use provided key or generate a default one
        self.key = key if key else "default_secret_key"
        # Create a hash of the key for consistent length
        self.key_hash = hashlib.sha256(self.key.encode()).digest()

    def encrypt_frame(self, frame):
        """
        Simple XOR encryption of frame data
        Args:
            frame: numpy array (video frame)
        Returns:
            encrypted_bytes: encrypted frame data
        """
        # Convert frame to bytes
        frame_bytes = frame.tobytes()
        
        # Create a key stream by repeating the hash
        key_stream = np.tile(np.frombuffer(self.key_hash, dtype=np.uint8), 
                           len(frame_bytes) // len(self.key_hash) + 1)
        key_stream = key_stream[:len(frame_bytes)]
        
        # XOR encryption
        encrypted = np.bitwise_xor(
            np.frombuffer(frame_bytes, dtype=np.uint8),
            key_stream
        )
        
        return encrypted.tobytes()

    def decrypt_frame(self, encrypted_data, frame_shape, frame_dtype):
        """
        Decrypt a video frame (XOR is symmetric, so decryption is the same as encryption)
        Args:
            encrypted_data: bytes (encrypted frame data)
            frame_shape: tuple (original frame shape)
            frame_dtype: numpy dtype (original frame dtype)
        Returns:
            frame: numpy array (decrypted frame)
        """
        # Create a key stream
        key_stream = np.tile(np.frombuffer(self.key_hash, dtype=np.uint8), 
                           len(encrypted_data) // len(self.key_hash) + 1)
        key_stream = key_stream[:len(encrypted_data)]
        
        # XOR decryption (same as encryption)
        decrypted = np.bitwise_xor(
            np.frombuffer(encrypted_data, dtype=np.uint8),
            key_stream
        )
        
        # Convert back to numpy array
        return decrypted.astype(frame_dtype).reshape(frame_shape)

    def get_key_base64(self):
        """Get the encryption key in base64 format for sharing"""
        return base64.b64encode(self.key.encode()).decode('utf-8')

    @classmethod
    def from_key_base64(cls, key_base64):
        """Create an instance from a base64 encoded key"""
        key = base64.b64decode(key_base64.encode('utf-8')).decode('utf-8')
        return cls(key) 