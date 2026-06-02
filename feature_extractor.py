import re
import socket
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import numpy as np

class FeatureExtractor:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.sensitive_words = ['login', 'secure', 'account', 'update', 'verify', 'bank', 'password', 'credential', 'signin']

    def _unshorten_url(self, url):
        try:
            # Add protocol if missing for the HEAD request
            test_url = url if url.startswith('http') else 'http://' + url
            response = requests.head(test_url, allow_redirects=True, timeout=2)
            return response.url
        except requests.RequestException:
            return url

    def extract_features(self, url):
        # Resolve shortlinks before any lexical parsing
        url = self._unshorten_url(url)
        
        # Initialize feature array with zeros
        features = np.zeros((1, len(self.feature_names)))
        
        # Add http:// if missing so urllib parses correctly
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
            
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ''
        path = parsed_url.path or ''
        query = parsed_url.query or ''
        
        # --- A. Lexical Features ---
        self._set_feature(features, 'UrlLength', len(url))
        self._set_feature(features, 'NumDots', url.count('.'))
        self._set_feature(features, 'NumDash', url.count('-'))
        self._set_feature(features, 'NumDashInHostname', hostname.count('-'))
        self._set_feature(features, 'AtSymbol', 1 if '@' in url else 0)
        self._set_feature(features, 'TildeSymbol', 1 if '~' in url else 0)
        self._set_feature(features, 'NumUnderscore', url.count('_'))
        self._set_feature(features, 'NumPercent', url.count('%'))
        self._set_feature(features, 'NumAmpersand', url.count('&'))
        self._set_feature(features, 'NumHash', url.count('#'))
        self._set_feature(features, 'NumNumericChars', sum(c.isdigit() for c in url))
        
        self._set_feature(features, 'NoHttps', 1 if not url.startswith('https://') else 0)
        self._set_feature(features, 'HttpsInHostname', 1 if 'https' in hostname else 0)
        self._set_feature(features, 'HostnameLength', len(hostname))
        self._set_feature(features, 'PathLength', len(path))
        self._set_feature(features, 'QueryLength', len(query))
        self._set_feature(features, 'DoubleSlashInPath', 1 if '//' in path else 0)
        
        # Subdomain & Path Level
        subdomains = hostname.split('.')
        self._set_feature(features, 'SubdomainLevel', len(subdomains) - 2 if len(subdomains) > 2 else 0)
        self._set_feature(features, 'PathLevel', path.count('/'))
        self._set_feature(features, 'NumQueryComponents', query.count('&') + 1 if query else 0)
        
        # Sensitive Words
        num_sensitive = sum(1 for word in self.sensitive_words if word in url.lower())
        self._set_feature(features, 'NumSensitiveWords', num_sensitive)
        
        # IP Address
        try:
            socket.inet_aton(hostname)
            is_ip = 1
        except socket.error:
            is_ip = 0
        self._set_feature(features, 'IpAddress', is_ip)
        
        # --- B. Content-Based Features ---
        html_content = ""
        try:
            # 3 second timeout for fast extraction
            response = requests.get(url, timeout=3, allow_redirects=True)
            if response.status_code == 200:
                html_content = response.text
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Iframe
                if soup.find_all('iframe') or soup.find_all('frame'):
                    self._set_feature(features, 'IframeOrFrame', 1)
                    
                # Missing Title
                if not soup.title or not soup.title.string:
                    self._set_feature(features, 'MissingTitle', 1)
                    
                # Insecure Forms
                forms = soup.find_all('form')
                insecure_forms = 0
                for form in forms:
                    action = form.get('action', '').lower()
                    if action.startswith('http://') or not action:
                        insecure_forms = 1
                self._set_feature(features, 'InsecureForms', insecure_forms)
                
                # Right Click Disabled (naive check)
                if 'event.button==2' in html_content or 'contextmenu' in html_content:
                    self._set_feature(features, 'RightClickDisabled', 1)
                    
        except Exception:
            pass # Keep defaults (0) if unreachable
            
        # --- C. RT / External Features (Heuristics) ---
        # UrlLengthRT (0 if short, 1 if suspicious)
        self._set_feature(features, 'UrlLengthRT', 1 if len(url) > 54 else 0)
        self._set_feature(features, 'SubdomainLevelRT', 1 if len(subdomains) > 3 else 0)
        
        return features
        
    def _set_feature(self, features_array, feature_name, value):
        if feature_name in self.feature_names:
            idx = self.feature_names.index(feature_name)
            features_array[0][idx] = value
