"""Streamlit only exposes page_title/page_icon via set_page_config; it
doesn't let you touch <head> directly. inject_head_metadata() reaches into
the parent document from a same-origin components.html iframe to set a real
lang attribute, meta description, and Open Graph tags (for link previews
when the app URL is shared).

Purely cosmetic/SEO — it does not affect and will not stop browser
extensions (e.g. Edge Copilot) from wrapping page content with their own
elements at read time; that happens client-side in the extension, after
this HTML is served.
"""

import streamlit.components.v1 as components


def inject_head_metadata():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.title = "Scout — Traced RAG";
        doc.documentElement.lang = "en";

        function setMeta(name, content, asProperty) {
            const attr = asProperty ? "property" : "name";
            let tag = doc.querySelector(`meta[${attr}="${name}"]`);
            if (!tag) {
                tag = doc.createElement("meta");
                tag.setAttribute(attr, name);
                doc.head.appendChild(tag);
            }
            tag.setAttribute("content", content);
        }

        setMeta("description",
            "A RAG research assistant that visualizes retrieval instead of hiding it \\u2014 see exactly which passages were selected and why.");
        setMeta("og:title", "Scout \\u2014 Traced RAG", true);
        setMeta("og:description",
            "A research assistant that shows its work: retrieval space, selection board, and cited answers.", true);
        setMeta("og:type", "website", true);
        </script>
        """,
        height=0, width=0,
    )