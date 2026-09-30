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