#!/usr/bin/env python3
"""Check crawlability and sitemap coverage in a Jekyll build (no network)."""

import argparse
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canonicals = []
        self.robots = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonicals.append(attrs.get("href"))
        if tag == "meta" and attrs.get("name", "").lower() in ("robots", "googlebot"):
            self.robots.extend(attrs.get("content", "").lower().replace(",", " ").split())


def check(build, origin):
    errors = []
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.parse(build / "sitemap.xml").getroot()
    if root.tag != "{http://www.sitemaps.org/schemas/sitemap/0.9}urlset":
        errors.append("Sitemap must be a namespaced urlset")
    entries = root.findall("s:url", namespace)
    urls = [entry.findtext("s:loc", default="", namespaces=namespace) for entry in entries]
    errors.extend(f"Duplicate sitemap URL: {url}" for url, n in Counter(urls).items() if n > 1)

    robots_text = (build / "robots.txt").read_text(encoding="utf-8")
    robots = RobotFileParser()
    robots.parse(robots_text.splitlines())
    if origin + "/sitemap.xml" not in (robots.site_maps() or []):
        errors.append("robots.txt does not advertise the canonical sitemap")

    expected = {origin + "/", origin + "/archives/"}
    for post in (build / "posts").rglob("index.html"):
        path = "/" + post.parent.relative_to(build).as_posix() + "/"
        metadata = Metadata()
        metadata.feed(post.read_text(encoding="utf-8"))
        if "noindex" not in metadata.robots and "none" not in metadata.robots:
            expected.add(origin + path)
    errors.extend(f"Missing sitemap URL: {url}" for url in sorted(expected - set(urls)))

    for entry, url in zip(entries, urls):
        parsed = urlsplit(url)
        if not url.startswith(origin + "/") or parsed.query or parsed.fragment:
            errors.append(f"Non-canonical sitemap URL: {url}")
            continue
        if not robots.can_fetch("Googlebot", url):
            errors.append(f"Googlebot is blocked by robots.txt: {url}")
        target = (build / unquote(parsed.path).lstrip("/")).resolve()
        if not target.is_relative_to(build.resolve()):
            errors.append(f"URL escapes build directory: {url}")
            continue
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            errors.append(f"Sitemap target does not exist: {url}")
            continue
        metadata = Metadata()
        metadata.feed(target.read_text(encoding="utf-8"))
        if metadata.canonicals != [url]:
            errors.append(f"Canonical mismatch: {url} -> {metadata.canonicals}")
        if "noindex" in metadata.robots or "none" in metadata.robots:
            errors.append(f"Sitemap URL has noindex: {url}")
        modified = entry.findtext("s:lastmod", namespaces=namespace)
        if modified:
            try:
                datetime.fromisoformat(modified.replace("Z", "+00:00"))
            except ValueError:
                errors.append(f"Invalid lastmod: {url} -> {modified}")

    return errors, len(urls), len(expected) - 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", nargs="?", type=Path, default=Path("_site"))
    parser.add_argument("--url", default="https://hahngyutak.github.io")
    args = parser.parse_args()
    try:
        errors, count, posts = check(args.build, args.url.rstrip("/"))
    except (OSError, ET.ParseError) as error:
        parser.exit(1, f"SEO check failed: {error}\n")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"SEO check passed: {posts} posts, {count} sitemap URLs; canonical URLs and robots rules valid.")


if __name__ == "__main__":
    main()
