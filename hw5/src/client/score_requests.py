import os

import requests
import rich


def main():
    url = os.getenv("API_URL")
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0,
    }
    rich.print(f"sending payload\n{client}\nto {url}")
    pred = requests.post(url, json=client).json()
    rich.print(f"resulting in\n", pred)


if __name__ == '__main__':
    main()
