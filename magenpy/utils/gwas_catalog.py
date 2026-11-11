import html
import re
from urllib.parse import unquote, urljoin, urlsplit

GWAS_CATALOG_SUMSTATS_ROOT = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
)


def is_gwas_catalog_accession(value):
    """Return whether ``value`` is a GWAS Catalog study accession."""

    return (
        isinstance(value, str)
        and re.fullmatch(r"GCST\d+", value.strip(), re.IGNORECASE) is not None
    )


def gwas_catalog_directory(accession):
    """Return the GWAS Catalog summary-statistics directory for an accession."""

    match = re.fullmatch(r"GCST(\d+)", str(accession).strip(), re.IGNORECASE)
    if match is None:
        raise ValueError("Enter a GWAS Catalog accession such as GCST005195.")

    number = int(match.group(1))
    if not 0 < number <= 2**53 - 1:
        raise ValueError("The GWAS Catalog accession is out of range.")

    width = max(6, len(match.group(1)))
    start = (number - 1) // 1000 * 1000 + 1
    study_id = f"GCST{number:0{width}d}"
    study_range = f"GCST{start:0{width}d}-GCST{start + 999:0{width}d}"
    return f"{GWAS_CATALOG_SUMSTATS_ROOT}{study_range}/{study_id}/"


def _listing_links(listing, directory):
    origin = urlsplit(directory)[:2]
    links = (
        urljoin(directory, html.unescape(match.group(1)))
        for match in re.finditer(r"href=[\"']([^\"']+)[\"']", listing, re.IGNORECASE)
    )
    return [
        url
        for url in links
        if urlsplit(url)[:2] == origin and url.startswith(directory)
    ]


def _is_build_37_filename(url):
    return (
        re.search(
            r"(?:grch37|build(?:grch)?37|hg19)",
            unquote(urlsplit(url).path),
            re.IGNORECASE,
        )
        is not None
    )


def _metadata_is_build_37(metadata):
    match = re.search(
        r"(?:^|\n)\s*genome_assembly\s*:\s*[\"']?([^\s#\"']+)",
        metadata,
        re.IGNORECASE,
    )
    return match is not None and match.group(1).lower() in {
        "grch37",
        "hg19",
        "ncbi37",
    }


def _read_text(url):
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, headers={"User-Agent": "magenpy"})) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding)
    except HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(
                "No GWAS Catalog summary statistics were found for this accession."
            ) from error
        raise ConnectionError(
            f"Could not list GWAS Catalog files (HTTP {error.code})."
        ) from error
    except URLError as error:
        raise ConnectionError(
            "Could not reach the GWAS Catalog FTP server. Check the connection."
        ) from error


def resolve_gwas_catalog(accession, read_text=None):
    """Resolve an accession to its preferred GRCh37 summary-statistics URL."""

    read_text = read_text or _read_text
    directory = gwas_catalog_directory(accession)
    root_links = _listing_links(read_text(directory), directory)
    candidates = []

    harmonised_directory = next(
        (url for url in root_links if url == directory + "harmonised/"), None
    )
    if harmonised_directory:
        links = _listing_links(read_text(harmonised_directory), harmonised_directory)
        for suffix in (".h.tsv.gz", ".f.tsv.gz"):
            candidates.extend(
                (url, links)
                for url in sorted(links)
                if urlsplit(url).path.lower().endswith(suffix)
            )

    candidates.extend(
        (url, root_links)
        for url in sorted(root_links)
        if re.search(r"\.tsv(?:\.gz)?$", urlsplit(url).path, re.IGNORECASE)
    )
    for candidate, links in candidates:
        if _is_build_37_filename(candidate):
            return candidate
        metadata_url = next(
            (url for url in links if url == candidate + "-meta.yaml"), None
        )
        if metadata_url:
            try:
                metadata = read_text(metadata_url)
            except Exception:
                continue
            if _metadata_is_build_37(metadata):
                return candidate

    raise ValueError(
        "This accession has no GRCh37/hg19 harmonised, formatted, or GWAS-SSF "
        "summary-statistics file."
    )
