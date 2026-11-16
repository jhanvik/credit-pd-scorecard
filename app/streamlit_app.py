"""Streamlit demo: score one credit-card customer with the scorecard and the LightGBM challenger.

Run from the project root:  uv run streamlit run app/streamlit_app.py
"""
import streamlit as st

from credit_pd_scorecard import config, inference

MONTHS = ["Sep 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005"]
PAY_OPTIONS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

st.set_page_config(page_title="Credit default scorecard demo", layout="wide")
st.title("Credit default (PD) scorecard demo")
st.caption(
    "Educational portfolio project. Trained on the UCI 'Default of Credit Card Clients' dataset (Taiwan, 2005). "
    "Illustrative only: not for real credit decisions. Cutoffs and business numbers rest on stated assumptions."
)

with st.form("customer"):
    limit = st.number_input("Credit limit (NT$)", min_value=1_000, value=200_000, step=10_000)
    st.markdown("**Monthly history** (month 1 = most recent, September 2005)")
    pay, bill, paid = [], [], []
    cols = st.columns(6)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{MONTHS[i]}**")
            pay.append(st.selectbox("Repayment status", PAY_OPTIONS, index=2, key=f"pay{i}",
                                    help="-1 paid duly; 1-9 months late. -2 and 0 are undocumented codes in the data."))
            bill.append(st.number_input("Bill (NT$)", value=30_000, step=1_000, key=f"bill{i}"))
            paid.append(st.number_input("Paid (NT$)", min_value=0, value=5_000, step=500, key=f"paid{i}"))
    submitted = st.form_submit_button("Score customer")

if submitted:
    raw = {"LIMIT_BAL": limit}
    raw.update(dict(zip(config.PAY_COLS, pay, strict=True)))
    raw.update(dict(zip(config.BILL_COLS, bill, strict=True)))
    raw.update(dict(zip(config.PAYAMT_COLS, paid, strict=True)))
    try:
        r = inference.score_customer(raw)
    except ValueError as err:
        st.error(f"Input problem: {err}")
    else:
        a, b, c = st.columns(3)
        a.metric("Scorecard PD", f"{r['pd_scorecard']:.1%}")
        b.metric("Score (higher = safer)", f"{r['score']:.0f}")
        c.metric("LightGBM PD (challenger)", f"{r['pd_lgbm']:.1%}")
        if r["flag_scorecard"]:
            st.warning("Scorecard flags this customer for the risk action (outreach / credit-limit review).")
        else:
            st.success("Scorecard does not flag this customer.")
        st.subheader("Main reasons (points below the average customer)")
        shown = [item for item in r["reasons"] if item["points_vs_average"] <= -0.5]  # ignore negligible ones
        if shown:
            for item in shown:
                st.write(f"- {item['reason']}: {item['points_vs_average']:.1f} points")
        else:
            st.write("No feature scores below the average customer.")
