import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Intelligence", layout="wide")
st.title("Document Intelligence RAG")

tab_ask, tab_docs = st.tabs(["კითხვა", "დოკუმენტები"])


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
    except requests.RequestException as e:
        st.error(f"API მიუწვდომელია: {e}")
        return None
    if r.status_code >= 400:
        st.error(f"შეცდომა {r.status_code}: {r.json().get('detail', r.text)}")
        return None
    return r.json()


with tab_ask:
    mode = st.radio("რეჟიმი", ["pipeline", "agent"], horizontal=True,
                    help="pipeline: ფიქსირებული ნაბიჯები. agent: LLM თავად იძახებს ინსტრუმენტებს.")
    question = st.text_area("თქვენი კითხვა (ინგლისურად)", height=100)
    if st.button("ჰკითხე", type="primary") and question.strip():
        with st.spinner("ვეძებ და ვაანალიზებ..."):
            data = api("POST", "/ask", json={"question": question, "mode": mode})
        if data:
            if data["abstained"]:
                st.warning("ინფორმაცია არასაკმარისია, პასუხს თავს ვიკავებ.")
            else:
                st.success(data["answer"])
            if data["conflict"]:
                st.error(f"ნაპოვნია წინააღმდეგობა: {data['conflict_note']}")
            if data["sub_queries"]:
                st.caption("ქვეკითხვები: " + " | ".join(data["sub_queries"]))
            if data["trace"]:
                with st.expander("აგენტის ნაბიჯები"):
                    for t in data["trace"]:
                        st.write(f"{t['step']}. `{t['tool']}` {t['args']}")
            st.subheader("წყაროები")
            for s in data["sources"]:
                with st.expander(f"[{s['number']}] {s['title']}"):
                    st.write(s["text"])
                    st.caption(f"chunk_id: {s['chunk_id']}")
            if not data["sources"]:
                st.caption("ციტირებული წყარო არ არის.")
            with st.expander(f"LLM-ს მიცემული ყველა პარაგრაფი ({len(data['retrieved'])})"):
                for s in data["retrieved"]:
                    st.markdown(f"**[{s['number']}] {s['title']}**")
                    st.write(s["text"][:400])

with tab_docs:
    st.subheader("ატვირთვა / განახლება")
    up = st.file_uploader("ფაილი", type=["md", "txt", "pdf", "docx", "jsonl"])
    if up and st.button("ატვირთვა"):
        with st.spinner("ვამუშავებ..."):
            res = api("POST", "/documents", files={"file": (up.name, up.getvalue())})
        if res:
            for r in res["results"]:
                st.success(f"{r['title']}: {r['status']} ({r['chunks']} chunk)")

    st.subheader("დოკუმენტები")
    q = st.text_input("ძიება სათაურით")
    page = st.number_input("გვერდი", min_value=1, value=1)
    size = 20
    data = api("GET", "/documents", params={"limit": size, "offset": (page - 1) * size, "q": q})
    if data:
        st.caption(f"სულ: {data['total']}")
        for d in data["items"]:
            c1, c2 = st.columns([6, 1])
            c1.write(f"**{d['title']}**  \n`{d['doc_id']}` · {d['format']} · {d['source']}")
            if c2.button("წაშლა", key=d["doc_id"]):
                if api("DELETE", f"/documents/{d['doc_id']}"):
                    st.rerun()