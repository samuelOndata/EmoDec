CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    image_url TEXT,
    predicted_label TEXT,
    correct_label TEXT,
    confidence FLOAT,
    is_correct BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);