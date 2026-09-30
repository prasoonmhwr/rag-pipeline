CREATE TABLE access_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE user_group_memberships (
    user_id UUID NOT NULL,
    group_id UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_id)
);

CREATE TABLE document_acl (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    group_id UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, group_id)
);

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunk_access_policy ON chunks
    USING (
        document_id IN (
            SELECT da.document_id FROM document_acl da
            JOIN user_group_memberships ugm ON ugm.group_id = da.group_id
            WHERE ugm.user_id = current_setting('app.current_user_id')::uuid
        )
    );

ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

-- A dedicated, unprivileged role for the running application to connect as.
-- This matters more than the policy itself: Postgres superusers and table owners
-- bypass Row-Level Security by default, no exceptions — and the POSTGRES_USER from
-- our docker-compose file (rag_user) is created as a superuser AND is the table
-- owner, since it's the role that ran schema.sql. If the app keeps connecting as
-- rag_user, every policy we just wrote is silently ignored, regardless of FORCE.
CREATE ROLE rag_app_user LOGIN PASSWORD 'change-this-in-production';
GRANT CONNECT ON DATABASE rag_db TO rag_app_user;
GRANT USAGE ON SCHEMA public TO rag_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON documents, chunks, access_groups, user_group_memberships, document_acl
    TO rag_app_user;

-- RLS defaults to deny-ALL commands once enabled, not just the command a policy
-- names. Our SELECT policy above only covers reads — without an explicit INSERT
-- policy too, rag_app_user's ingestion writes into chunks would be rejected
-- outright the moment it stops being a superuser. This grants that unconditionally,
-- since ingestion is a trusted backend process, not something we gate per-user.
CREATE POLICY chunk_insert_policy ON chunks
    FOR INSERT
    WITH CHECK (true);