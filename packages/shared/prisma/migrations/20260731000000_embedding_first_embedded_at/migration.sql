-- Drift windows were built from embeddings.created_at, which the embedding worker moves
-- every time it re-encodes a document. Activating an adapted model re-encodes the entire
-- corpus, so a backfill restamped every historical vector into the current window: the
-- window then covered the whole corpus instead of recent arrivals, and any drift score
-- computed during a backfill was meaningless.
--
-- first_embedded_at records when a document first became searchable and is never updated
-- afterwards, so drift measures arrival while created_at continues to track the latest
-- write and therefore backfill progress.
ALTER TABLE "embeddings"
    ADD COLUMN "first_embedded_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- Existing rows have only ever been written once, so their creation time is also the point
-- at which they first became searchable.
UPDATE "embeddings" SET "first_embedded_at" = "created_at";

CREATE INDEX "embeddings_first_embedded_at_idx" ON "embeddings" ("first_embedded_at");
