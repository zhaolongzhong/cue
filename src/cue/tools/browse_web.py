import logging
from typing import Union

import requests
from bs4 import BeautifulSoup

from .utils.search import duckduckgo_search, search_news

logger: logging.Logger = logging.getLogger(__name__)

commands = {
    "search": "search",
    "open_url": "open_url",
    "news": "news",
}


def browse_web(command: str, args: list[str]) -> Union[str, list[any]]:
    logger.debug(f"browse_web: {command}, {args}")
    if command == commands["search"]:
        return search(args)  # return a list search results with title, url, and snippet
    elif command == commands["open_url"]:  # return web page contents
        return [open_url(url) for url in args]
    elif command == commands["news"]:
        return news(args)  # return a list news
    else:
        return "Command not found."


def search(args: list[str]) -> Union[str, list[any]]:
    combined_query = " ".join(args)
    try:
        text = duckduckgo_search(combined_query)
        # text = google_search(combined_query)
    except Exception as e:
        logger.error(f"Error fetching with web_search for {combined_query}: {e}")
        text = f"Error fetching the page. {e}"
    return text


def news(args: list[str]) -> Union[str, list[any]]:
    combined_query = " ".join(args)
    results = search_news(combined_query)
    if not results:
        return "No news found. Try search."
    return results


def open_url(url: str) -> str:
    """Fetches and returns the text content of a specified web page.

    This function sends a request to the given URL and extracts the text content
    of the page using BeautifulSoup. It is designed to work with pages that render
    their content in HTML and might not work with pages that rely heavily on JavaScript
    for rendering content.

    Args:
        url (str): The URL of the web page to retrieve.

    Returns:
        str: The text content of the web page, or None if an error occurred or JavaScript is required.
    """
    # Reference: https://github.com/openai/openai-cookbook/blob/9e09df530dbf02c050e4dfff5e4f8e4eb35a26ac/apps/web-crawl-q-and-a/web-qa.py
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            # pylint: disable=broad-exception-raised
            # raise Exception(f"Error fetching {url}: HTTP {response.status_code}")
            logger.error(f"Error fetching {url}: HTTP {response.status_code}")
            return f"Error fetching the page. HTTP {response.status_code}"

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract and return the text content
        # text = soup.get_text()  # only text
        # text with hyperlinks and hyperlinks
        text, _hyperlinks = extract_text_and_links(soup)
        # print(f"hyperlinks: {hyperlinks}")

        # Check for JavaScript requirement
        if "You need to enable JavaScript to run this app." in text:
            logger.warning("Unable to parse page due to JavaScript being required")
            return "Unable to parse page due to JavaScript being required"
        text = remove_newlines(text)

        # Calculate visibility percentage
        total_length = len(text)
        max_response_length = 20000
        if total_length > max_response_length:
            visible_text = text[:max_response_length]
            visible_length = len(visible_text)
            visibility_percentage = (visible_length / total_length) * 100
            text = f"{visible_text}\nVisible: 0% - {visibility_percentage:.2f}%"

        title = soup.title.string if soup.title else ""
        # Attempt to extract the publish date
        publish_date = find_publish_date(soup)

        result = {
            "title": title,
            "url": url,
            "pub_date": publish_date if publish_date else "",
            "text": text,
        }
        return result

    except Exception as e:
        logger.error("An error occurred: %s", e)
        return "Cannot open page."


def find_publish_date(soup):
    # Common meta tags used for publish date
    date_meta_tags = [
        {"name": "pubdate"},
        {"name": "publishdate"},
        {"name": "DC.date.issued"},
        {"name": "date"},
        {"property": "article:published_time"},
    ]
    try:
        for tag in date_meta_tags:
            date_tag = soup.find("meta", attrs=tag)
            if date_tag and date_tag.get("content"):
                publish_date = date_tag["content"]
                break

        # Another common location for publish date
        if not publish_date:
            publish_date = soup.find("time")
            if publish_date:
                publish_date = publish_date.get("datetime", publish_date.text)
    except Exception as e:
        logger.warn("Error finding publish date: %s", e)
        publish_date = None

    return publish_date


def remove_newlines(text):
    text = text.replace("\n", " ")
    text = text.replace("\\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    return text


def extract_text_and_links(soup):
    texts = []
    hyperlinks = []
    for element in soup.find_all(["p", "a"]):
        if element.name == "a" and "href" in element.attrs:
            link = element["href"].strip()
            if link:  # Filter out empty strings
                texts.append(f"{element.get_text()} ({link})")
                hyperlinks.append(link)
        else:
            texts.append(element.get_text())
    return " ".join(texts), hyperlinks


if __name__ == "__main__":
    browse_open_url_result = browse_web("open_url", ["https://en.wikipedia.org/wiki/Artificial_intelligence"])
    browse_search_result = browse_web("search", ["Today's major AI news"])
    print(f"browse open_url:\n{browse_open_url_result}")
    print(f"browse search:\n{browse_search_result}")
