/* ───────────────────────── HYDRATE QUOTES (DEPTH ≤ 3) ───────────────────────── */
WITH RECURSIVE
/* 0. conversation root --------------------------------------------------------- */
base AS (
    SELECT *
    FROM   nindexer.casts
    WHERE  root_parent_hash =
           decode('e7126eb73294f4d3014f657d2de26dce4203d0bb','hex')
),

/* 1. crawl embedded_casts ≤ 3, break cycles via path --------------------------- */
rec AS (
    SELECT  b.id, b.hash, b.text,
            b.embedded_urls, b.embedded_casts,
            b.mentions,      b.mentions_positions,
            b.fid,
            0                       AS lvl,
            ARRAY[b.hash]::bytea[]  AS path
    FROM    base b
    UNION ALL
    SELECT  c.id, c.hash, c.text,
            c.embedded_urls, c.embedded_casts,
            c.mentions,      c.mentions_positions,
            c.fid,
            r.lvl + 1,
            r.path || c.hash
    FROM    rec r
    CROSS   JOIN LATERAL (
              SELECT DISTINCT h.hash
              FROM   unnest(r.embedded_casts) AS h(hash)
            ) h
    JOIN    nindexer.casts c ON c.hash = h.hash
    WHERE   r.lvl < 3
      AND   array_position(r.path, c.hash) IS NULL
),

/* 2. safe @handle injection (earliest offset per handle keeps) ---------------- */
pos AS (
    SELECT DISTINCT ON (r.id, tag)
           r.id,
           length(
             convert_from(
               substring(convert_to(r.text,'utf8')
                         FROM 1 FOR r.mentions_positions[i]),
               'utf8'))            AS char_pos,
           tag
    FROM   rec r
    CROSS  JOIN LATERAL generate_subscripts(r.mentions,1) AS gs(i)
    JOIN   profiles p ON p.fid = r.mentions[gs.i]
    CROSS  JOIN LATERAL (SELECT '@'||p.username AS tag) t
    ORDER  BY r.id, tag, char_pos
),
seg AS (
    SELECT  p.id,
            p.char_pos,
            substr(r.text,
                   COALESCE(lag(p.char_pos) OVER w,0)+1,
                   p.char_pos-COALESCE(lag(p.char_pos) OVER w,0))
            || p.tag                AS chunk
    FROM    pos p
    JOIN    rec r USING (id)
    WINDOW  w AS (PARTITION BY p.id ORDER BY p.char_pos)
),
rebuild AS (
    SELECT id,
           string_agg(chunk,'' ORDER BY char_pos) AS head,
           max(char_pos)                          AS last_char
    FROM   seg
    GROUP  BY id
),

/* 3. URLs not in text ---------------------------------------------------------- */
missing_urls AS (
    SELECT  r.id,
            array_agg(DISTINCT url) AS urls
    FROM    rec r
    CROSS   JOIN unnest(r.embedded_urls) AS u(url)
    WHERE   position(lower(url) IN lower(r.text)) = 0
    GROUP  BY r.id
),

/* 4. text body + URL tail ------------------------------------------------------ */
core AS (
    SELECT  r.id,
            r.hash,
            COALESCE(head || substr(r.text,last_char+1), r.text)  AS body,
            COALESCE(
              CASE WHEN mu.urls IS NULL OR array_length(mu.urls,1)=0
                   THEN '' ELSE ' '||array_to_string(mu.urls,' ') END,'')
                                                               AS tail,
            p.username                                           AS author,
            r.embedded_casts
    FROM    rec r
    LEFT    JOIN rebuild      USING (id)
    LEFT    JOIN missing_urls mu USING (id)
    JOIN    profiles p ON p.fid = r.fid
),

/* 5. bottom-up: ONE fully-hydrated text per cast (no duplicate bubbling) ----- */
quote_tree AS (
    /* leaves */
    SELECT  c.id,
            c.hash,
            c.body || c.tail                    AS full_text,
            c.author
    FROM    core c
    WHERE   c.embedded_casts IS NULL
       OR   array_length(c.embedded_casts,1)=0

    UNION ALL
    /* parents */
    SELECT  c.id,
            c.hash,
            c.body
            || ' ' || format(
                   'QUOTE:["%s" - @%s]',
                   qt.full_text,
                   qt.author)
            || c.tail                            AS full_text,
            c.author
    FROM    core c
    CROSS   JOIN LATERAL (
              SELECT DISTINCT h.hash
              FROM   unnest(c.embedded_casts) AS h(hash)
            ) h
    JOIN    quote_tree qt ON qt.hash = h.hash
),

/* 6. final hydrated text + @dup guard --------------------------------------- */
final_text AS (
    SELECT DISTINCT ON (id)
           id,
           regexp_replace(full_text,
              '(@[A-Za-z0-9._]+)\1+', '\1', 'g') AS text_populated
    FROM   quote_tree
    ORDER  BY id
)

/* 7. project one row per root-thread cast ----------------------------------- */
SELECT  b.id,
        COALESCE(ft.text_populated, b.text) AS text_populated
FROM    base b
LEFT    JOIN final_text ft USING (id)
ORDER  BY b.id;


