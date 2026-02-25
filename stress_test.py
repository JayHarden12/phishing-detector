from services.classification import ClassificationService
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def run_stress_test():
    service = ClassificationService()
    config = {
        'naive_bayes': {'upper': 0.7, 'lower': 0.35},
        'svm': {'upper': 0.7, 'lower': 0.35},
        'logistic_regression': {'upper': 0.7, 'lower': 0.35}
    }
    
    test_urls = [
        "https://google.com",
        "http://www.secure-update-login.account-verify-support.com/auth/login.php?id=9a8b7c",
        "Just a normal string",
        "",
        "http://" + "a" * 500 + ".com",  # Very long URL
        "http://[2001:db8::1]/path", # IPv6
        "javascript:alert(1)" # JS execution attempt
    ]
    
    print("--- STRESS TEST REPORT ---")
    for url in test_urls:
        print(f"\nTesting URL: {url[:50]}...")
        try:
            res = service.classify(url, "all", config)
            if "error" in res:
                print(f"PASS (Handled Error): {res['error']}")
            else:
                print(f"PASS (Success): Pred {res['prediction']}, Prob {res['phishing_probability']:.2f}")
        except Exception as e:
            print(f"FAIL: Hard crash - {str(e)}")

if __name__ == "__main__":
    run_stress_test()
