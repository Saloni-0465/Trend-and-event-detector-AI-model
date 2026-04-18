import pandas as pd

from src.events import label_events


def test_label_events_adds_label_and_category():
    # Two time bins with categories that clearly dominate.
    test_df = pd.DataFrame(
        {
            "time_bin": [pd.Timestamp("2018-05-03", tz="UTC"), pd.Timestamp("2018-05-10", tz="UTC")],
            "category": ["POLITICS", "TECH"],
        }
    )

    events_out = {
        "lexical": [
            {
                "event_id": 1,
                "start_bin": pd.Timestamp("2018-05-03", tz="UTC"),
                "end_bin": pd.Timestamp("2018-05-10", tz="UTC"),
                "score": 0.364,
                "drivers": {"new_terms": ["house", "white"], "dropped_terms": ["trump"]},
            }
        ],
        "lda": [
            {
                "event_id": 1,
                "start_bin": pd.Timestamp("2018-06-21", tz="UTC"),
                "end_bin": pd.Timestamp("2018-06-28", tz="UTC"),
                "score": 0.083,
                "drivers": {
                    "top_topics": [
                        {"topic_id": 3, "words": ["trump", "women", "said"]},
                        {"topic_id": 4, "words": ["new", "black"]},
                    ]
                },
            }
        ],
    }

    labeled = label_events(events_out, test_df, include_category=True)

    lex = labeled["lexical"][0]
    assert "label" in lex
    assert "Lexical change" in lex["label"]
    # Dominant category within the window [start_bin, end_bin] is tie-broken deterministically.
    assert lex["dominant_category"] in {"POLITICS", "TECH"}

    lda = labeled["lda"][0]
    assert "Topic-mixture change" in lda["label"]
    # For LDA event window bins are outside test_df range => no matching category.
    assert lda["dominant_category"] is None

