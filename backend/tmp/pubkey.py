from Crypto.PublicKey import RSA

key = """
-----BEGIN PGP PUBLIC KEY BLOCK-----
mDMEXfTzshYJKwYBBAHaRw8BAQdAMKRu+4d3VND4fzNAyMoQi3ja3dqDnBhw2z4p
sDXvnhK0Qi9yYXcgUE9ORyBfR0hNb2FDWExUIChxOWYpIDw1ODg4MzQwMytxOWZA
dXNlcnMubm9yZXBseS5naXRodWIuY29tPoiQBBMWCAA4FiEEtXtdHtJ17C6DWoUh
55zfoXHcLfkFAl3087ICGwMFCwkIBwIGFQoJCAsCBBYCAwECHgECF4AACgkQ55zf
oXHcLflgoAD+J7yAeA8l+1CfbjKQQ7yJ48Bb3YqmR/yJi4fXHJjTDV0A/0JjzE8h
1sskdGy+H6ab5CI5hb39TZe6UcBJp4AJE7sBuDgEXfTzshIKKwYBBAGXVQEFAQEH
QBDtzmNJntzK78qhwwUkbj7Q/03SrNch8TphQ46/SyMRAwEIB4h4BBgWCAAgFiEE
tXtdHtJ17C6DWoUh55zfoXHcLfkFAl3087ICGwwACgkQ55zfoXHcLfkkWwEA+yYQ
714WmPLIW5quzJ1X9M9AZGXAHLLdF8PEVKTVZzABAM1HUAK2xAUL1WDpdWjWc+td
zErdlysAV+qcQf8BVYAO=b+gd
-----END PGP PUBLIC KEY BLOCK-----
""".replace("\n","")
# RSA.importKey(key)
print(key)
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_PKCS1_v1_5
from base64 import b64decode, b64encode

pubkey = 'MIGfMA0GCSqGSIb3DQEBA3UAA4GNADCBiQKBgQC35eMaYoJXEoJt5HxarHkzDBEMU3qIWE0HSQ77CwP/8UbX07W2XKwngUyY4k6Hl2M/n9TOZMZsiBzer/fqV+QNPN1m9M94eUm2gQgwkoRj5battRCaNJK/23GGpCsTQatJN8PZBhJBb2Vlsvw5lFrSdMT1R7vaz+2EeNR/FitFXwIDAQAB'
msg = "test"
keyDER = b64decode(pubkey)
keyPub = RSA.importKey(keyDER)
cipher = Cipher_PKCS1_v1_5.new(keyPub)
cipher_text = cipher.encrypt(msg.encode())
emsg = b64encode(cipher_text)
print(emsg)
