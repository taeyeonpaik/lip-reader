"""Inventory or download an authenticated TCD-TIMIT directory recursively.

Standard-library only. A browser session Cookie header is entered privately at
the terminal; it is never stored or printed. No login/CAPTCHA automation.
Default is inventory only. --download downloads the chosen subtree.
"""
import argparse
import getpass
import fcntl
import threading
import time
import re
import zipfile
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.parse import urljoin, urlsplit, unquote
from urllib.request import Request, build_opener, HTTPRedirectHandler

ORIGIN = 'https://sigmedia.tcd.ie'
ROOT = '/tcd_timit_db/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.directories = set()
        self._href = None
        self._label = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)
                self._href = href
                self._label = []

    def handle_data(self, data):
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag):
        if tag == 'a' and self._href is not None:
            if ''.join(self._label).strip().endswith('/'):
                self.directories.add(self._href)
            self._href = None
            self._label = []


class SameOrigin(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != 'https' or urlsplit(newurl).netloc != 'sigmedia.tcd.ie':
            raise RuntimeError('Refusing off-site redirect with authenticated session')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def relative(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.netloc != 'sigmedia.tcd.ie':
        return None
    path = unquote(parsed.path)
    if not path.startswith(ROOT) or parsed.query or parsed.fragment:
        return None
    rel = PurePosixPath(path[len(ROOT):])
    if '..' in rel.parts or rel.is_absolute():
        return None
    return str(rel)


def frontal_selection(url):
    """Select one archive per speaker, never its duplicate extracted camera tree."""
    rel = relative(url)
    if rel is None:
        return False
    parts = PurePosixPath(rel).parts
    if not parts:
        return True
    if len(parts) == 1 and parts[0] in ('README.txt', 'volunteer_labelfiles.mlf', 'lipspeaker_labelfiles.mlf'):
        return True
    if parts[0] not in ('volunteers', 'lipspeakers'):
        return False
    if any(p in ('straightcam', '30degcam') for p in parts):
        return False
    if urlsplit(url).path.endswith('/'):
        return True
    return parts[-1] == 'straightcam.zip'


def directory_target(parent, href, labelled_directory=False):
    target = urljoin(parent, href)
    rel = relative(target)
    if rel is None:
        return target
    parts = PurePosixPath(rel).parts
    # The portal can display "volunteers/" but link to "volunteers".
    # Also recognize extensionless directories inside the two recording trees.
    known_directory = (parts and parts[0] in ('volunteers', 'lipspeakers')
                       and not PurePosixPath(parts[-1]).suffix)
    if labelled_directory or known_directory:
        return target.rstrip('/') + '/'
    return target


def incomplete_selection(selected, report):
    """Retry unfinished entries only; completed entries retain their receipts."""
    states = report.get('files', {})
    return [url for url in selected
            if states.get(relative(url), {}).get('state') not in ('verified', 'complete')]




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=ORIGIN + ROOT)
    parser.add_argument('--all-files', action='store_true', help='Disable frontal archive selection; may download duplicate camera data')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 9), help='Concurrent downloads (1–8)')
    parser.add_argument('--from-inventory', action='store_true', help='Reuse the previously saved inventory')
    parser.add_argument('--retry-incomplete', action='store_true', help='Select only unfinished files from the previous progress report; do not rehash completed files')
    parser.add_argument('--retry-attempts', type=int, default=6, help='Maximum attempts per file')
    parser.add_argument('--retry-base-seconds', type=int, default=5)
    parser.add_argument('--retry-max-seconds', type=int, default=120)
    parser.add_argument('--max-archives', type=int, help='Limit this download to N archives, volunteers first; reference files always included')
    args = parser.parse_args()
    if not 1 <= args.retry_attempts <= 100 or args.retry_base_seconds < 1 or args.retry_max_seconds < args.retry_base_seconds:
        parser.error('Use 1–100 attempts and positive retry delays with max >= base')
    if args.retry_incomplete and not (args.from_inventory and args.download):
        parser.error('--retry-incomplete requires --from-inventory --download')
    if args.max_archives is not None and args.max_archives < 1:
        parser.error('--max-archives must be positive')
    if relative(args.url) is None:
        parser.error('Use an HTTPS directory URL within the official TCD-TIMIT database')
    # Directory-relative links must resolve beneath this path, not its parent.
    start = args.url.rstrip('/') + '/'
    cookie = getpass.getpass('Paste the Cookie request-header VALUE from your signed-in browser (hidden): ').strip()
    if not cookie or '\n' in cookie or '\r' in cookie:
        raise ValueError('Expected one nonempty Cookie header value')

    def request(url, extra_headers=None):
        headers = {'Cookie': cookie, 'User-Agent': 'TCD-research-downloader/1.0', 'Accept-Encoding': 'identity'}
        headers.update(extra_headers or {})
        response = build_opener(SameOrigin()).open(Request(url, headers=headers), timeout=90)
        if '/users/' in urlsplit(response.url).path:
            response.close()
            raise RuntimeError('Session expired or not accepted. Sign in in your browser again.')
        return response

    inventory = args.destination / 'inventory.json'
    if args.from_inventory:
        files = set(json.loads(inventory.read_text()))
        if not all(relative(u) is not None and u.startswith(start) and
                   (args.all_files or frontal_selection(u)) for u in files):
            raise RuntimeError('Saved inventory contains out-of-scope URLs')
    else:
        queue, visited, files, diagnostics = [start], set(), set(), []
        while queue:
            url = queue.pop()
            if url in visited:
                continue
            visited.add(url)
            with request(url) as response:
                page = response.read(8 * 1024 * 1024).decode('utf-8')
            if 'user[password]' in page:
                raise RuntimeError('Login required; no recordings downloaded')
            links = Links()
            links.feed(page)
            diagnostics.append(dict(directory=url, links=[h for h in links.links
                               if relative(urljoin(url, h)) is not None]))
            for href in links.links:
                target = directory_target(url, href, href in links.directories)
                if relative(target) is None or not target.startswith(start):
                    continue
                if not args.all_files and not frontal_selection(target):
                    continue
                if urlsplit(target).path.endswith('/'):
                    queue.append(target)
                else:
                    files.add(target)
            print(f'Listed {len(visited)} directories; {len(files)} files', flush=True)
        if not files:
            raise RuntimeError('No files found; check directory URL and session')
        args.destination.mkdir(parents=True, exist_ok=True)
        inventory = args.destination / 'inventory.json'
        inventory.write_text(json.dumps(sorted(files), indent=2) + '\n')
        (args.destination / 'inventory-directories.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        print(f'Inventory saved: {inventory}')
    if not args.all_files and not any(url.endswith('/straightcam.zip') for url in files):
        raise RuntimeError('No frontal archives found. Directory diagnostics saved; no media downloaded.')
    if not args.download:
        print('No media downloaded. Inspect inventory and select the desired subtree before using --download.')
        return
    root = args.destination.resolve()
    selected = sorted(files, key=lambda u: (u.endswith('/straightcam.zip'), '/lipspeakers/' in u, u))
    if args.max_archives is not None:
        references = [u for u in selected if not u.endswith('/straightcam.zip')]
        archives = [u for u in selected if u.endswith('/straightcam.zip')]
        selected = references + archives[:args.max_archives]
    with (root / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.retry_incomplete:
            report = json.loads((root / 'download-progress.json').read_text())
            # Preserve the original all-file report before the retry run replaces it.
            history = root / 'download-progress-before-retry.json'
            if not history.exists():
                history.write_text(json.dumps(report, indent=2)+'\n')
            previous = json.loads(history.read_text())
            previous['files'].update(report.get('files', {}))
            # Keep successful retries across repeated invocations, too.
            tmp = history.with_suffix('.tmp')
            tmp.write_text(json.dumps(previous, indent=2)+'\n')
            tmp.replace(history)
            selected = incomplete_selection(selected, previous)
            print(f'Retrying {len(selected)} unfinished files; completed files are retained.', flush=True)
        delays = tuple(min(args.retry_max_seconds, args.retry_base_seconds * 2**i)
                       for i in range(args.retry_attempts - 1))
        raise SystemExit('Inventory complete; transfer support is introduced in the next milestone')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped; completed files remain verified.', file=sys.stderr)
        sys.exit(130)
