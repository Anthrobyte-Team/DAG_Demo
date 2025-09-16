import os, re, hashlib, functools, requests, html, fitz, uuid
import logging
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl, urljoin
import streamlit as st
from bs4 import BeautifulSoup
from .sql_store import url_by_file
from ImageAgent.sql_store import get_imageagent_sources
from ImageAgent.router import pick_best_folder_name

logger = logging.getLogger(__name__)

# Config
MAX_PDF_PAGES_TO_SHOW = 4
WEB_SNIPPET_MAX_CHARS  = 700
PDF_SNIPPET_MAX_CHARS  = 700
PREFER_WEB_IF_PRESENT  = False
TOP_WEB_RESULTS        = 1

ALLOWED_TAGS  = {"p","ul","ol","li","a","b","strong","i","em","code","pre","h1","h2","h3","h4","h5","h6","blockquote"}
ALLOWED_ATTRS = {"a": {"href","title","target","rel"}}
_HTML_CACHE: dict[str, str] = {}

# Utility: headers / sanitization
def _ua() -> str:
    return "RAG-App/1.0 (+contact@example.com)"

def _sanitize_html(html_str: str) -> str:
    try:
        soup = BeautifulSoup(html_str or "", "lxml")
        for tag in soup.find_all(True):
            if tag.name not in ALLOWED_TAGS:
                tag.unwrap()
            else:
                tag.attrs = {k: v for k, v in (tag.attrs or {}).items()
                             if (tag.name in ALLOWED_ATTRS and k in ALLOWED_ATTRS[tag.name])}
        return str(soup)
    except Exception:
        return ""

def _absolutize_links(html_str: str, base_url: str) -> str:
    try:
        soup = BeautifulSoup(html_str or "", "lxml")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if href.startswith(("javascript:", "mailto:", "tel:")):
                continue
            a["href"] = urljoin(base_url, href)
            a["target"] = "_blank"
            a["rel"] = "noopener noreferrer"
        return str(soup)
    except Exception:
        return html_str or ""

# ---------------- Web fetch / extract ----------------
def _fetch_main_html(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": _ua()}, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def _extract_article_html(raw_html: str, url: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    for t in soup(["script","style","noscript"]): t.decompose()
    if "wikipedia.org" in (url or ""):
        main = soup.select_one("#mw-content-text")
    else:
        main = soup.select_one("article") or soup.select_one("main") or soup.body
    return str(main) if main else raw_html

def _load_main_html(url: str) -> str:
    if url in _HTML_CACHE:
        return _HTML_CACHE[url]
    raw = _fetch_main_html(url)
    main = _extract_article_html(raw, url)
    _HTML_CACHE[url] = main
    return main

# ---------------- Text helpers ----------------
_STOP = set("""
a an and the of for in on to with from as by or if is are were was be been being it that this
those these they them he she you your i we our us their its not no at into over under out up down
which who whom whose what when where why how can could should would will
about explain tell info information brief overview details retrieved archive archived doi issn isbn http https www com org net
""".split())

def _strip_citations_only(txt: str) -> str:
    # Remove numeric [1], [2] style citations but KEEP URLs
    txt = re.sub(r"\[\s*\d+\s*\]", "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

def _linkify(text: str) -> str:
    """Convert raw URLs inside text to clickable <a> links (keeps everything else escaped)."""
    if not text:
        return ""
    escaped = html.escape(text, quote=True)
    return re.sub(
        r'(https?://[^\s<>()]+)',
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>',
        escaped
    )

def _is_noisy_paragraph(p: str) -> bool:
    p_low = p.lower()
    if "http://" in p_low or "https://" in p_low: return True
    if "doi.org" in p_low or "issn" in p_low or "isbn" in p_low: return True
    if len(re.findall(r"\[\d+\]", p)) >= 1: return True
    if re.match(r"^\s*(?:•|-|\d{1,3}[\.\)])\s", p): return True
    if len(p.strip()) < 60: return True
    letters = sum(ch.isalpha() for ch in p)
    return letters < max(1, int(0.55 * len(p)))

def _is_reference_page(text: str) -> bool:
    t = (text or "")
    low = t.lower()
    head = "\n".join(low.splitlines()[:12])
    if any(h in head for h in ["references","bibliography","external links","further reading","citations","notes", "table of contents","contents","index","appendix"]):
        return True
    url_count  = low.count("http://") + low.count("https://")
    doi_count  = len(re.findall(r"\bdoi[:/\.]", low))
    isbn_count = len(re.findall(r"\bisbn\b", low))
    issn_count = len(re.findall(r"\bissn\b", low))
    enum_lines = sum(1 for ln in low.splitlines() if re.match(r"\s*(?:•|-|\d{1,3}[\.\)])\s", ln))
    if (isbn_count + issn_count + doi_count) >= 2:
        return True
    if url_count >= 3 and enum_lines >= 2:
        return True
    letters = sum(ch.isalpha() for ch in t)
    return letters < max(1, int(0.55 * len(t)))

def _best_paragraph(text: str, question: str, max_chars: int = 700) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()] or [text.strip()]
    clean = [p for p in paras if not _is_noisy_paragraph(p)] or paras
    q_toks = [w for w in re.findall(r"[a-z0-9]+", (question or "").lower()) if w not in _STOP]
    def overlap(p): return sum(1 for w in q_toks if w in p.lower())
    snip = max(clean, key=overlap) if q_toks else clean[0]
    snip = _strip_citations_only(snip)
    return (snip[:max_chars] + "…") if len(snip) > max_chars else snip

def _match_score(text: str, question: str) -> int:
    if not text or not question: return 0
    toks_q = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOP]
    toks_t = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not toks_q or not toks_t: return 0
    set_t = set(toks_t)
    uni = sum(1 for w in toks_q if w in set_t)
    big_q = {" ".join([toks_q[i], toks_q[i+1]]) for i in range(len(toks_q)-1)}
    big_t = {" ".join([toks_t[i], toks_t[i+1]]) for i in range(len(toks_t)-1)}
    bi = len(big_q & big_t)
    score = uni + 2 * bi
    if _is_noisy_paragraph(text):
        score = max(0, score - 2)
    return score

# ---------------- PDF: stream from Blob (no local files) ----------------
@functools.lru_cache(maxsize=16)
def _pdf_bytes(url: str) -> bytes:
    if not url:
        return b""
    r = requests.get(url, headers={"User-Agent": _ua()}, timeout=20)
    r.raise_for_status()
    return r.content

def _open_doc_from_url(url: str):
    if not (fitz and url):
        return None
    data = _pdf_bytes(url)
    if not data:
        return None
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception:
        return None

def _page_text(doc, page_idx: int) -> str:
    try:
        if 0 <= page_idx < len(doc):
            return doc.load_page(page_idx).get_text("text")
    except Exception:
        pass
    return ""

def _subset_pdf_bytes(doc, pages: list[int]) -> bytes:
    if not doc or not pages:
        return b""
    try:
        out = fitz.open()
        for p in pages:
            if 0 <= p < len(doc):
                out.insert_pdf(doc, from_page=p, to_page=p)
        data = out.write()
        out.close()
        return bytes(data)
    except Exception:
        return b""

def _open_page_link(file_url: str, page_idx: int) -> str:
    return f"{file_url}#page={page_idx+1}" if file_url else ""

# ---------------- Annotation-aware inline link helpers ----------------
def _linkify_pdf_text_from_page(page, text: str) -> str:
    if not page:
        return ""
    links = page.get_links() or []
    replacements = []
    for lk in links:
        if "uri" not in lk:
            continue
        try:
            t = page.get_textbox(lk["from"]).strip()
        except Exception:
            t = ""
        if t and len(t) > 2 and (t, lk["uri"]) not in replacements:
            replacements.append((t, lk["uri"]))

    def _safe_once(s, needle, uri):
        pat = r'(?<!\w){}(?!\w)'.format(re.escape(needle))
        repl = f'<a href="{html.escape(uri)}" target="_blank" rel="noopener noreferrer">{html.escape(needle)}</a>'
        return re.sub(pat, repl, s, count=1)

    out = html.escape(text or "")
    for t, uri in replacements:
        out = _safe_once(out, html.escape(t), uri)  # escape needle to match escaped text
    return "<p>" + out + "</p>"

def _pdf_snippet_html_from_doc(doc, page_idx: int, para: str) -> str:
    if not doc or not (0 <= page_idx < len(doc)) or not para:
        return ""
    page = doc.load_page(page_idx)
    return _linkify_pdf_text_from_page(page, para)

def _pdf_page_html_from_doc(doc, page_idx: int) -> str:
    if not doc or not (0 <= page_idx < len(doc)):
        return ""
    page = doc.load_page(page_idx)
    text = page.get_text("text") or ""
    blocks = [b.strip() for b in re.split(r"\n{2,}|\r\n{2,}", text) if len(b.strip()) > 30] or [text.strip()]
    return "\n".join(_linkify_pdf_text_from_page(page, b) for b in blocks if b)

# ---------------- Web helpers ----------------
def _canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if not k.lower().startswith("utm_") and k.lower() not in {"ref","ref_src"}]
        params.sort()
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))
    except Exception:
        return url

def _render_snippet_with_links(url: str, text_snippet: str) -> str | None:
    html_main = _load_main_html(url)
    if not html_main or not text_snippet:
        return None
    soup = BeautifulSoup(html_main, "lxml")
    def norm(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()
    s_norm = norm(text_snippet)
    target = None
    for p in soup.select("p"):
        text = norm(p.get_text(" ", strip=True))
        if s_norm[:60] and s_norm[:60] in text:
            target = p
            break
    para_html = target.decode_contents() if target else None
    return _sanitize_html(para_html) if para_html else None

def _render_context_block(url: str, text_snippet: str, before=2, after=2) -> str | None:
    """Return sanitized HTML containing the nearest heading + nearby paragraphs around the best-matching paragraph."""
    html_main = _load_main_html(url)
    if not html_main or not text_snippet:
        return None
    soup = BeautifulSoup(html_main, "lxml")

    # find target <p> that contains the start of our snippet
    def norm(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()
    s_norm = norm(text_snippet)[:80]
    target = None
    for p in soup.select("p"):
        if s_norm and s_norm in norm(p.get_text(" ", strip=True)):
            target = p
            break
    if not target:
        return None

 # collect nearest headings (any level) + N paragraphs before/after
    block = []

    # headings above target
    prev_heads = target.find_all_previous(["h1","h2","h3","h4","h5","h6"])
    prev_heads = list(reversed(prev_heads))  # keep document order

    # headings after target (useful when paragraph is in the lead section)
    next_heads = target.find_all_next(["h1","h2","h3","h4","h5","h6"])

    # pick up to two closest headings: prefer previous, else next
    picked = (prev_heads[-2:] if prev_heads else next_heads[:2])
    for h in picked:
        block.append(h)


    prevs, nxts = [], []
    x = target
    while x and len(prevs) < before:
        x = x.find_previous("p")
        if x: prevs.append(x)
    x = target
    while x and len(nxts) < after:
        x = x.find_next("p")
        if x: nxts.append(x)

    for p in reversed(prevs): block.append(p)
    block.append(target)
    for p in nxts: block.append(p)

    frag = BeautifulSoup("<div></div>", "lxml").div
    for el in block:
        frag.append(BeautifulSoup(str(el), "lxml"))
    html = _absolutize_links(str(frag), url)
    return _sanitize_html(html)

def find_imageagent_pdf_for(question: str) -> dict | None:
    """Return {'title','pdf_url','description'} for the best ImageAgent PDF, or None."""
    if not get_imageagent_sources or not pick_best_folder_name:
        return None
    rows = [
        r for r in get_imageagent_sources()
        if r.get("file_type") == "images" and r.get("agent_type") == "imageagent"
    ]
    folder_names = [r.get("file_name") for r in rows if isinstance(r.get("file_name"), str)]
    if not folder_names:
        return None
    try:
        best = pick_best_folder_name(question, folder_names)
    except Exception:
        best = folder_names[0]
    row = next((r for r in rows if r.get("file_name") == best), None)
    if not row or not row.get("blob_url"):
        return None
    return {
        "title": row.get("file_name") or "ImageAgent PDF",
        "pdf_url": row["blob_url"],
        "description": row.get("description", "PDF generated by ImageAgent.")
    }


# Main renderer
def render_sources(question: str, docs, key_seed: str = ""):
    if not docs:
        return

    web_seen, web_items = set(), []
    pdf_groups: dict[tuple[str, str], dict] = {}
    imageagent_pdfs = []

    def _ensure_pdf_group(name: str, url: str):
        key = (name, url)
        if key not in pdf_groups:
            pdf_groups[key] = {"pages": set(), "name": name, "url": url}
        return key

    # Group candidates
    for d in docs:
        meta = d.metadata or {}
        # --- ImageAgent PDF special handling ---
        if meta.get("type") == "imageagent_pdf":
            imageagent_pdfs.append(meta)
            continue
        # --- Normal RAG logic ---
        stype = (meta.get("source_type") or "").lower()
        page  = meta.get("page")
        file_url = (meta.get("file_url") or "").strip()

        is_pdf = (stype == "pdf")
        is_web = (stype == "url") or str(meta.get("canonical_url") or meta.get("source_url") or meta.get("source") or "").startswith(("http://","https://"))

        if is_web and not is_pdf:
            url = meta.get("canonical_url") or meta.get("source_url") or meta.get("source") or ""
            url = _canonical_url(url)
            if url and url not in web_seen:
                web_seen.add(url)
                title = (meta.get("title") or url or "Web page").strip()
                web_items.append({
                    "title": title,
                    "url": url,
                    "snippet": _best_paragraph(d.page_content, question, WEB_SNIPPET_MAX_CHARS),
                    "score": _match_score(d.page_content, question),
                })
        elif is_pdf:
            name = (meta.get("display_name") or meta.get("pdf_title") or meta.get("file_name") or "document.pdf").strip()
            url  = file_url or url_by_file(meta.get("file_id"), meta.get("file_name"))
            if url and isinstance(page, int) and page >= 0:
                _ensure_pdf_group(name, url)
                pdf_groups[(name, url)]["pages"].add(page)
        else:
            url = _canonical_url(str(meta.get("source") or ""))
            if url and url not in web_seen:
                web_seen.add(url)
                title = (meta.get("title") or url or "Web page").strip()
                web_items.append({
                    "title": title,
                    "url": url,
                    "snippet": _best_paragraph(d.page_content, question, WEB_SNIPPET_MAX_CHARS),
                    "score": _match_score(d.page_content, question),
                })

    # Keep top web hit and never drop PDFs unless PREFER_WEB_IF_PRESENT
    web_items = sorted(web_items, key=lambda x: x.get("score", 0), reverse=True)[:TOP_WEB_RESULTS]
    if PREFER_WEB_IF_PRESENT and web_items:
        pdf_groups = {}

    with st.expander("Sources", expanded=False):
        # --- Rank and trim PDF groups by best page score, keep top N ---
        MAX_PDF_DOCS_TO_SHOW = 4
        ranked = []
        for (name, url), g in list(pdf_groups.items()):
            doc = _open_doc_from_url(url)
            if not doc or not g["pages"]:
                continue
            page_texts = {p: _page_text(doc, p) for p in g["pages"]}
            page_scores = {p: _match_score(page_texts.get(p, ""), question) for p in g["pages"]}
            best = max(page_scores.values() or [0])
            ranked.append(((name, url), best))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top_keys = {k for k, _ in ranked[:MAX_PDF_DOCS_TO_SHOW]}
        pdf_groups = {k: v for k, v in pdf_groups.items() if k in top_keys}

        #  Web sources
        if web_items:
            st.markdown("Web sources")
            it = web_items[0]
            st.markdown(f"1. [{it['title']}]({it['url']})")
            html_para = _render_snippet_with_links(it["url"], it["snippet"])
            if html_para:
                st.markdown(_sanitize_html(_absolutize_links(html_para, it["url"])), unsafe_allow_html=True)
            else:
                st.markdown(_linkify(it["snippet"]), unsafe_allow_html=True)

            with st.expander("Show web page extract"):
                context_html = _render_context_block(it["url"], it["snippet"], before=2, after=2)
                if context_html:
                    st.markdown(context_html, unsafe_allow_html=True)
                else:
                    st.info("Context unavailable for this page.")

        # PDF sources (RAG)
        if pdf_groups:
            st.markdown("PDF sources")
            for (name, url), g in pdf_groups.items():
                pages_seen = sorted(g["pages"])
                if not pages_seen:
                    continue

                doc = _open_doc_from_url(url)
                if not doc:
                    with st.expander(f"{name} — pages: n/a"):
                        st.info("Couldn’t open this PDF in the app.")
                        st.markdown(f"[Open PDF]({url})")
                    continue

                # score pages vs. question
                page_texts = {p: _page_text(doc, p) for p in pages_seen}
                page_scores = {p: _match_score(page_texts.get(p, ""), question) for p in pages_seen}
                max_sc = max(page_scores.values() or [0])
                thr = max(1, int(0.50 * max_sc))
                filtered = [p for p in pages_seen if page_scores.get(p, 0) >= thr and not _is_reference_page(page_texts.get(p, ""))]
                if not filtered:
                    filtered = sorted(pages_seen, key=lambda p: page_scores.get(p, 0), reverse=True)[:MAX_PDF_PAGES_TO_SHOW]

                pages_to_show = sorted(filtered, key=lambda p: page_scores.get(p, 0), reverse=True)[:MAX_PDF_PAGES_TO_SHOW]
                label = ", ".join(str(p + 1) for p in pages_to_show) if pages_to_show else "n/a"

                with st.expander(f"{name} — pages: {label}"):
                    key_prefix = hashlib.md5(f"{question}|{name}|{url}".encode()).hexdigest()[:8]
                    selected_pages = []

                    for p in pages_to_show:
                        st.markdown(f"Page {p+1}")
                        full_txt = page_texts.get(p, "") or ""

                        paras = [para.strip() for para in re.split(r"\n{2,}|\r\n{2,}", full_txt) if len(para.strip()) > 35] or [full_txt.strip()]
                        best_para = _best_paragraph("\n\n".join(paras), question, PDF_SNIPPET_MAX_CHARS)

                        if best_para.strip():
                            html_snip = _pdf_snippet_html_from_doc(doc, p, _strip_citations_only(best_para))
                            st.markdown(html_snip if html_snip else "(no extractable text)", unsafe_allow_html=True)
                        else:
                            st.write("(no extractable text)")

                        link = _open_page_link(url, p)
                        if link:
                            st.markdown(f"[Open page {p+1}]({link})")

                        checked = st.checkbox("Include in download", value=True, key=f"sel_{key_prefix}_{p}")
                        if checked:
                            selected_pages.append(p)

                        # Only for RAG PDFs: show full page text
                        with st.expander("Show full page text"):
                            html_full = _pdf_page_html_from_doc(doc, p)
                            st.markdown(html_full if html_full else "(text unavailable)", unsafe_allow_html=True)
                        st.divider()

                    final_pages = selected_pages or pages_to_show
                    data = _subset_pdf_bytes(doc, final_pages)
                    st.download_button(
                    label="Download selected pages (PDF)",
                    data=data,
                    file_name=f"{os.path.splitext(name)[0]}-pages-{'-'.join(str(p+1) for p in final_pages)}.pdf" if final_pages else f"{os.path.splitext(name)[0]}-pages.pdf",
                    mime="application/pdf",
                    disabled=(not data),
                    key=f"ragpdf_{key_prefix}_{uuid.uuid4()}"
                )


        # ImageAgent PDFs
        # if imageagent_pdfs:
        #     st.markdown("PDFs generated by ImageAgent")
        #     for meta in imageagent_pdfs:
        #         title = meta.get("title") or meta.get("source") or "ImageAgent PDF"
        #         pdf_url = meta.get("source")
        #         st.markdown(f"[{title}]({pdf_url})")
        #         # st.write(meta.get("description", "PDF generated by ImageAgent."))
        #         # Download button
        #         try:
        #             response = requests.get(pdf_url)
        #             if response.status_code == 200:
        #                 st.download_button(
        #                     label="Download PDF",
        #                     data=response.content,
        #                     file_name=title.replace(" ", "_") + ".pdf",
        #                     mime="application/pdf",
        #                     key=f"imageagentpdf_{hashlib.md5(pdf_url.encode()).hexdigest()}_{uuid.uuid4()}"  # <-- Add unique key here
        #                 )
        #             else:
        #                 st.error("Failed to fetch PDF from blob storage.")
        #         except Exception as e:
        #             st.error(f"Error fetching PDF: {e}")

def filter_sources_by_answer(docs, answer: str, question: str):
    """Keep only the docs that best support the final answer."""
    scored = []
    for d in docs:
        text = d.page_content or ""
        score_q = _match_score(text, question)
        score_a = _match_score(text, answer)
        score = score_a * 2 + score_q
        scored.append((score, d))
    if not scored:
        return docs
    max_sc = max(s for s, _ in scored)
    keep = [d for s, d in scored if s >= 0.55 * max_sc]
    return keep or [max(scored, key=lambda x: x[0])[1]]

from langchain.schema import Document

def append_imageagent_source(docs, question: str):
    """Return docs + a synthetic Document for the best ImageAgent PDF (if any)."""
    meta = find_imageagent_pdf_for(question)
    if not meta:
        return docs
    img_doc = Document(
        page_content="",
        metadata={
            "type": "imageagent_pdf",
            "title": meta["title"],
            "source": meta["pdf_url"],
            "description": meta["description"],
        },
    )
    return list(docs) + [img_doc]


def build_relevant_pdf(docs, user_msg: str) -> bytes | None:
    from PyPDF2 import PdfMerger
    import io
    merger = PdfMerger()
    kept = False

    pages_by_url = {}
    for d in docs:
        meta = d.metadata or {}

        # NEW: if it's the ImageAgent PDF, append the whole file
        if meta.get("type") == "imageagent_pdf":
            try:
                url = meta.get("source") or meta.get("pdf_url") or ""
                if url:
                    content = _pdf_bytes(url)  # uses requests under the hood
                    if content:
                        merger.append(io.BytesIO(content))
                        kept = True
            except Exception:
                pass
            continue

        # Normal RAG PDF pages
        if (meta.get("source_type") == "pdf" or isinstance(meta.get("page"), int)):
            url = (meta.get("file_url") or "").strip()
            if url:
                pages_by_url.setdefault(url, set()).add(meta.get("page"))

    # Append selected pages per RAG PDF
    for url, pages in pages_by_url.items():
        doc = _open_doc_from_url(url)
        if not doc:
            continue
        page_texts = {p: _page_text(doc, p) for p in pages}
        scores = {p: _match_score(page_texts.get(p, ""), user_msg) for p in pages}
        if not scores:
            continue
        max_sc = max(scores.values())
        thr = max(1, int(0.5 * max_sc))
        filtered = [p for p in pages if scores[p] >= thr and not _is_reference_page(page_texts[p])]
        if not filtered:
            filtered = sorted(scores, key=scores.get, reverse=True)[:MAX_PDF_PAGES_TO_SHOW]
        subset = _subset_pdf_bytes(doc, sorted(filtered))
        if subset:
            merger.append(io.BytesIO(subset))
            kept = True

    if kept:
        out = io.BytesIO()
        merger.write(out)
        merger.close()
        return out.getvalue()
    return None