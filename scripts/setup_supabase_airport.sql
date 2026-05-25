-- ============================================================
-- Supabase Schema: Airport AI Security System
-- Chạy file này trong Supabase SQL Editor (Project → SQL Editor)
-- ============================================================

-- ── 1. Bảng airport_events (lịch sử alert, persistent) ──────
CREATE TABLE IF NOT EXISTS airport_events (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  event_type    TEXT        NOT NULL,
  -- Giá trị: 'abandoned_baggage' | 'weapon_detected'

  camera_id     TEXT        NOT NULL DEFAULT 'CAM_00',
  track_id      INTEGER,
  -- ByteTrack ID của object/người bị phát hiện

  object_class  TEXT,
  -- COCO: 'suitcase' | 'backpack' | 'handbag' | 'knife' | 'gun'

  confidence    FLOAT       CHECK (confidence >= 0 AND confidence <= 1),
  risk_level    TEXT        DEFAULT 'high',
  -- Giá trị: 'low' | 'medium' | 'high' | 'critical'

  zone_name     TEXT,
  -- Tên khu vực (VD: 'Gate B3', 'Security Checkpoint')

  duration_sec  FLOAT       DEFAULT 0,
  -- Hành lý: số giây bỏ lại; Vũ khí: 0

  image_url     TEXT,
  -- Public URL ảnh snapshot trên Supabase Storage

  metadata      JSONB       DEFAULT '{}',
  -- Dữ liệu mở rộng: {"bbox": [x1,y1,x2,y2], "bearer_id": null}

  resolved      BOOLEAN     DEFAULT FALSE,
  -- True khi nhân viên an ninh đã xử lý

  resolved_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_airport_events_type
  ON airport_events(event_type);

CREATE INDEX IF NOT EXISTS idx_airport_events_camera
  ON airport_events(camera_id);

CREATE INDEX IF NOT EXISTS idx_airport_events_created
  ON airport_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_airport_events_resolved
  ON airport_events(resolved);

CREATE INDEX IF NOT EXISTS idx_airport_events_risk
  ON airport_events(risk_level);

-- Row Level Security
ALTER TABLE airport_events ENABLE ROW LEVEL SECURITY;

-- Tạo policy (dùng DO block để tránh lỗi nếu đã tồn tại)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'airport_events'
      AND policyname = 'airport_events_service_role'
  ) THEN
    CREATE POLICY "airport_events_service_role"
      ON airport_events FOR ALL USING (true);
  END IF;
END $$;


-- ── 2. Bảng baggage_tracks (realtime state, upsert liên tục) ─
CREATE TABLE IF NOT EXISTS baggage_tracks (
  track_id       INTEGER     PRIMARY KEY,
  -- ByteTrack object ID (duy nhất theo session)

  camera_id      TEXT        NOT NULL DEFAULT 'CAM_00',
  object_class   TEXT        NOT NULL,
  -- 'suitcase' | 'backpack' | 'handbag'

  first_seen_at  TIMESTAMPTZ DEFAULT NOW(),
  last_seen_at   TIMESTAMPTZ DEFAULT NOW(),
  -- Tự động cập nhật khi upsert

  has_owner      BOOLEAN     DEFAULT TRUE,
  -- False khi chủ đã rời đi

  owner_gone_at  TIMESTAMPTZ,
  -- Thời điểm chủ rời đi (NULL nếu đang có chủ)

  alerted        BOOLEAN     DEFAULT FALSE,
  -- True khi đã gửi alert cho track này

  bbox           JSONB,
  -- {"x1": 100, "y1": 200, "x2": 300, "y2": 400}

  updated_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-update updated_at khi có UPDATE
CREATE OR REPLACE FUNCTION update_baggage_tracks_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_baggage_tracks_updated_at ON baggage_tracks;
CREATE TRIGGER trg_baggage_tracks_updated_at
  BEFORE UPDATE ON baggage_tracks
  FOR EACH ROW
  EXECUTE FUNCTION update_baggage_tracks_updated_at();

-- RLS
ALTER TABLE baggage_tracks ENABLE ROW LEVEL SECURITY;

-- Tạo policy (dùng DO block để tránh lỗi nếu đã tồn tại)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE tablename = 'baggage_tracks'
      AND policyname = 'baggage_tracks_service_role'
  ) THEN
    CREATE POLICY "baggage_tracks_service_role"
      ON baggage_tracks FOR ALL USING (true);
  END IF;
END $$;


-- ── 3. Enable Realtime cho baggage_tracks ────────────────────
-- Bật trong Supabase Dashboard:
--   Database → Replication → baggage_tracks → Enable
--
-- Hoặc chạy lệnh sau (cần quyền superuser):
-- ALTER PUBLICATION supabase_realtime ADD TABLE baggage_tracks;


-- ── 4. Xác nhận kết quả ──────────────────────────────────────
SELECT
  table_name,
  (SELECT COUNT(*) FROM information_schema.columns c
   WHERE c.table_name = t.table_name
     AND c.table_schema = 'public') AS col_count
FROM information_schema.tables t
WHERE table_schema = 'public'
  AND table_name IN ('airport_events', 'baggage_tracks')
ORDER BY table_name;
