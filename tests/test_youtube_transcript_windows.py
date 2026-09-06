"""The windowed (`?timestamps=1`) variant of the YouTube transcript endpoint.

Three layers, because the shape is only useful if all three hold:

1. the formatter itself (fixed-width cues, absolute buckets, one window per
   bucket in time order),
2. the route (the default answer stays the plain string it has always been; the
   variant is opt-in and echoes the mode it resolved),
3. the retrieval half — the windowed markdown pushed through the real
   ``MarkdownHeadingSplitter`` gives EVERY chunk a ``[HH:MM:SS]`` heading, which
   is the only reason a `###` heading was chosen over a bare bracketed line.

All transcript text here is invented. Nothing in this file is a real caption
track.
"""
import re

import pytest
from fastapi.testclient import TestClient

from knowledge_api_server import app


def _seg(start, text, duration=2.0):
    return {"start": start, "duration": duration, "text": text}


#: Three 120-second windows' worth of invented captions, deliberately not
#: aligned to the boundaries: 119.5 belongs to window 0, 120.0 opens window 1.
FIXTURE_SEGMENTS = [
    _seg(0.0, "welcome to the walkthrough"),
    _seg(4.0, "we start with the index layout"),
    _seg(119.5, "and that is the layout"),
    _seg(120.0, "now the query path"),
    _seg(241.0, "finally the evaluation harness"),
]


class TestFormatTranscriptWindows:
    def _fmt(self, segments, **kw):
        from main.fetchers.youtube.youtube_transcript_downloader import format_transcript_windows
        return format_transcript_windows(segments, **kw)

    def test_windows_are_headings_with_fixed_width_cues(self):
        out = self._fmt(FIXTURE_SEGMENTS)
        assert out == (
            "### [00:00:00]\n"
            "welcome to the walkthrough we start with the index layout and that is the layout\n"
            "\n"
            "### [00:02:00]\n"
            "now the query path\n"
            "\n"
            "### [00:04:00]\n"
            "finally the evaluation harness"
        )

    def test_hour_boundary_is_zero_padded_to_eight_characters(self):
        # `_format_timestamp`'s MM:SS/HH:MM:SS switch is exactly what this must
        # NOT do: a variable-width cue makes the heading unparseable downstream.
        # 3721 s floors to the 3720 s window = 01:02:00 — hours AND minutes
        # non-zero, so the padding of both fields is under test.
        out = self._fmt([_seg(3721.0, "an hour in"), _seg(59.0, "the first minute")])
        assert out.splitlines()[0] == "### [00:00:00]"
        assert "### [01:02:00]" in out
        for line in out.splitlines():
            if line.startswith("### "):
                assert re.fullmatch(r"### \[\d{2}:\d{2}:\d{2}\]", line), line

    def test_bucket_is_the_floor_of_the_window(self):
        # 239.999 is still window 2 (120..240); 240.0 opens window 3.
        out = self._fmt([_seg(239.999, "still window two"), _seg(240.0, "window three")])
        assert "### [00:02:00]\nstill window two" in out
        assert "### [00:04:00]\nwindow three" in out

    def test_segment_exactly_on_a_boundary_opens_the_new_window(self):
        out = self._fmt([_seg(120.0, "on the boundary")])
        assert out == "### [00:02:00]\non the boundary"

    def test_out_of_order_segments_yield_one_window_per_bucket_in_time_order(self):
        out = self._fmt([
            _seg(130.0, "second window, first arrival"),
            _seg(5.0, "first window"),
            _seg(125.0, "second window, second arrival"),
        ])
        assert out == (
            "### [00:00:00]\nfirst window\n"
            "\n"
            "### [00:02:00]\nsecond window, first arrival second window, second arrival"
        )
        assert out.count("### [00:02:00]") == 1

    def test_blank_segments_produce_no_window(self):
        out = self._fmt([_seg(0.0, "real text"), _seg(200.0, "   "), _seg(400.0, "")])
        assert out == "### [00:00:00]\nreal text"

    def test_no_segments_is_empty_string(self):
        assert self._fmt([]) == ""

    def test_newlines_inside_a_segment_are_normalised_to_single_spaces(self):
        # YouTube caption cues carry hard line breaks; those are layout, and a
        # raw newline inside a window body would read as a new markdown block.
        out = self._fmt([_seg(0.0, "first line\nsecond   line")])
        assert out == "### [00:00:00]\nfirst line second line"

    def test_negative_start_folds_into_the_first_window(self):
        out = self._fmt([_seg(-3.0, "before zero"), _seg(1.0, "at zero")])
        assert out == "### [00:00:00]\nbefore zero at zero"
        assert out.count("### [") == 1

    def test_custom_window_width(self):
        out = self._fmt([_seg(0.0, "a"), _seg(61.0, "b")], window_sec=60)
        assert out == "### [00:00:00]\na\n\n### [00:01:00]\nb"

    @pytest.mark.parametrize("bad", [0, -60, float("inf"), float("nan")])
    def test_window_sec_must_be_finite_and_positive(self, bad):
        # `match=` is load-bearing: without the guard, inf and nan still raise a
        # ValueError — from `math.floor(nan)`, several lines later and with a
        # message about float conversion. A bare `raises(ValueError)` therefore
        # passes on unguarded code and pins nothing for those two.
        with pytest.raises(ValueError, match="window_sec must be a finite number"):
            self._fmt(FIXTURE_SEGMENTS, window_sec=bad)

    def test_plain_formatter_is_unchanged(self):
        from main.fetchers.youtube.youtube_transcript_downloader import YouTubeTranscriptDownloader
        assert YouTubeTranscriptDownloader.format_transcript_plain(None, FIXTURE_SEGMENTS) == (
            "welcome to the walkthrough we start with the index layout "
            "and that is the layout now the query path finally the evaluation harness"
        )


class _FakeDownloader:
    """Stands in for the network half of ``fetch_transcript``."""

    def __init__(self, *args, **kwargs):
        pass

    def download_transcript(self, video_id):
        return {"available": True, "language": "en", "segments": FIXTURE_SEGMENTS}

    def format_transcript_plain(self, segments):
        from main.fetchers.youtube.youtube_transcript_downloader import YouTubeTranscriptDownloader
        return YouTubeTranscriptDownloader.format_transcript_plain(self, segments)


def _fake_downloader(segments):
    """A ``_FakeDownloader`` subclass serving one specific segment list."""

    class _Fixed(_FakeDownloader):
        def download_transcript(self, video_id):
            return {"available": True, "language": "en", "segments": segments}

    return _Fixed


PLAIN_EXPECTED = (
    "welcome to the walkthrough we start with the index layout "
    "and that is the layout now the query path finally the evaluation harness"
)


class TestYouTubeTranscriptRoute:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        import main.ingest.youtube as yt
        monkeypatch.setattr(yt, "YouTubeTranscriptDownloader", _FakeDownloader)

    def _get(self, params=None):
        return TestClient(app).get("/api/youtube/transcript/abcdefghijk", params=params or {})

    def test_default_response_is_the_plain_string(self):
        body = self._get().json()
        assert body["video_id"] == "abcdefghijk"
        assert body["transcript"] == PLAIN_EXPECTED
        assert body["char_count"] == len(PLAIN_EXPECTED)
        assert body["timestamps"] is False
        # The keys muninn's two summarizers read, plus the additive echo. A key
        # dropped here is a silent break in another repo.
        assert set(body) == {"video_id", "transcript", "char_count", "timestamps"}

    def test_timestamps_1_returns_the_windowed_shape(self):
        body = self._get({"timestamps": "1"}).json()
        assert body["timestamps"] is True
        assert body["transcript"].startswith("### [00:00:00]\n")
        assert "### [00:02:00]\nnow the query path" in body["transcript"]
        assert body["char_count"] == len(body["transcript"])

    def test_timestamps_0_is_the_default_response(self):
        off = self._get({"timestamps": "0"}).json()
        assert off == self._get().json()
        assert off["timestamps"] is False
        assert off["transcript"] == PLAIN_EXPECTED

    def test_timestamps_true_word_is_accepted(self):
        assert self._get({"timestamps": "true"}).json()["timestamps"] is True

    def test_unparseable_timestamps_value_is_a_422(self):
        assert self._get({"timestamps": "maybe"}).status_code == 422

    def test_valueless_timestamps_is_a_422(self):
        # `?timestamps=` is FastAPI's bool_parsing 422, and it stays one: a
        # valueless flag silently meaning true is the worse answer, since the
        # windowed body is not what an existing caller parses.
        response = TestClient(app).get("/api/youtube/transcript/abcdefghijk?timestamps=")
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "bool_parsing"

    def test_a_non_numeric_segment_start_is_a_422(self, monkeypatch):
        # Every other bad input on this route is a 422; a track whose `start`
        # is not a number must not be the one that is a 500.
        import main.ingest.youtube as yt
        monkeypatch.setattr(
            yt, "YouTubeTranscriptDownloader",
            _fake_downloader([{"start": "abc", "duration": 2.0, "text": "hello"}]),
        )
        response = self._get({"timestamps": "1"})
        assert response.status_code == 422
        assert "abc" in response.json()["detail"]

    def test_the_timestamps_parameter_is_documented(self):
        # The route's only reader is another repo; the OpenAPI description is
        # where the flag is discoverable, and every other optional flag in
        # `main/routes/` carries one.
        schema = TestClient(app).get("/openapi.json").json()
        params = schema["paths"]["/api/youtube/transcript/{video_id}"]["get"]["parameters"]
        described = {p["name"]: p.get("description", "") for p in params}
        assert "windowed" in described["timestamps"].lower()


class TestWindowedTranscriptThroughTheSplitter:
    """The retrieval half: every chunk of a windowed transcript names its window.

    Same property ``TestVimeoDocumentThroughTheConverter`` pins for Vimeo — with
    bare bracketed lines only the chunks that happen to START on a boundary
    carry a cue, so a hit in a long talk cannot be cited to the minute.
    """

    _CUES = ("[00:00:00]", "[00:02:00]", "[00:04:00]")
    _TOKEN_RE = re.compile(r"w(\d)s\d\d")

    def _segments(self):
        segments = []
        for window in range(1, 4):
            base = (window - 1) * 120
            for i in range(1, 26):
                segments.append(_seg(
                    base + i * 4,
                    f"w{window}s{i:02d} the speaker keeps explaining the retrieval pipeline.",
                ))
        return segments

    def test_every_chunk_names_its_window(self):
        from main.fetchers.youtube.youtube_transcript_downloader import format_transcript_windows
        from main.sources.files.markdown_heading_splitter import MarkdownHeadingSplitter

        markdown = format_transcript_windows(self._segments())
        chunks = MarkdownHeadingSplitter(chunk_size=1000, chunk_overlap=100).split(markdown)

        seen = {}
        for chunk in chunks:
            windows = {int(m) for m in self._TOKEN_RE.findall(chunk["text"])}
            assert windows, "a chunk with no window token means the fixture drifted"
            assert len(windows) == 1, "a chunk spans two windows; the cue is ambiguous"
            cue = self._CUES[windows.pop() - 1]
            assert chunk["heading"] == cue
            seen[cue] = seen.get(cue, 0) + 1

        assert set(seen) == set(self._CUES)
        # Each window is longer than chunk_size, so this pins the MID-SECTION
        # chunk and not only the one that starts on the heading line.
        assert all(count >= 2 for count in seen.values()), seen
