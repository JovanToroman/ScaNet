def insert_into_db(image_name, status, pipeline_phase, ssim_orb, ssim_ransac_flow, phone, conn):
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO image VALUES (?, ?, ?, ?, ?, ?)",
                  (image_name, status, pipeline_phase, ssim_orb, ssim_ransac_flow, phone))
    except Exception as e:
        print(e)
    conn.commit()


def check_if_image_exists(image_name, pipeline_phase, phone, conn):
    c = conn.cursor()
    c.execute('SELECT * FROM image WHERE name=? AND pipeline_phase=? AND phone=?', (image_name, pipeline_phase, phone))
    res = c.fetchone()
    return res and len(res) > 0


