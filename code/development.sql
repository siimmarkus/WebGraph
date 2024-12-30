SELECT visit_id, request_id, url, headers, top_level_url, resource_type,
    time_stamp, post_body, post_body_raw from http_requests where 3035506289898499 = visit_id;

SELECT visit_id, request_id, url, headers, response_status, time_stamp, content_hash
    from http_responses where 3035506289898499 = visit_id;

SELECT visit_id, old_request_id, old_request_url, new_request_url, response_status,
            headers, time_stamp from http_redirects where 3035506289898499 = visit_id;

SELECT visit_id, request_id, call_stack from callstacks where 3035506289898499 = visit_id;

SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,
            arguments, attributes, value, time_stamp from javascript where 3035506289898499 = visit_id;


CREATE INDEX visit_id_http_requests_index ON http_requests (visit_id);
CREATE INDEX visit_id_http_responses_index ON http_responses (visit_id);
CREATE INDEX visit_id_http_redirects_index ON http_redirects (visit_id);
CREATE INDEX visit_id_callstacks_index ON callstacks (visit_id);
CREATE INDEX visit_id_javascript_index ON javascript (visit_id);


SELECT visit_id, site_url from site_visits where visit_id in (
    SELECT visit_id from crawl_history where command = 'GetCommand' and command_status = 'ok'
    ) ORDER BY RANDOM() LIMIT 1;

SELECT visit_id from crawl_history where command = 'GetCommand' and command_status = 'ok';

SELECT COUNT(*) FROM javascript;

SELECT COUNT(*) FROM http_responses WHERE content_hash IS NOT NULL;

CREATE EXTENSION HSTORE;
CREATE TABLE leveldb (h hstore);
INSERT INTO leveldb VALUES ('a=>b, c=>d');
SELECT h['c'] FROM leveldb;

DROP TABLE IF EXISTS key_value;
CREATE TABLE IF NOT EXISTS key_value (
    key TEXT,
    value TEXT
);

select
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))),
  pg_total_relation_size(quote_ident(table_name))
from information_schema.tables
where table_schema = 'public'
order by 3 desc;