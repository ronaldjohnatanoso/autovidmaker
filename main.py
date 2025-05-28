import json
from playwright.sync_api import sync_playwright

def fix_cookie(cookie):
    # Rename expirationDate -> expires (Playwright wants seconds since epoch)
    if "expirationDate" in cookie:
        cookie["expires"] = float(cookie.pop("expirationDate"))
    else:
        cookie["expires"] = -1  # Means no expiry

    # Fix sameSite values
    same_site_map = {
        "lax": "Lax",
        "strict": "Strict",
        "no_restriction": "None",
        "none": "None"
    }
    s = cookie.get("sameSite", "").lower()
    cookie["sameSite"] = same_site_map.get(s, "Lax")  # Default to Lax if unknown

    # Remove unsupported keys
    for key in ["hostOnly", "session", "storeId", "id"]:
        cookie.pop(key, None)

    # Make sure only expected keys remain
    allowed_keys = {"name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite"}
    return {k: v for k, v in cookie.items() if k in allowed_keys}

def main():
    with open("cookies.json", "r") as f:
        raw_cookies = json.load(f)

    fixed_cookies = [fix_cookie(c) for c in raw_cookies]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        context.add_cookies(fixed_cookies)
        page = context.new_page()
        page.goto("https://meta.ai")
        input("Press Enter to exit...")
        browser.close()

if __name__ == "__main__":
    main()
