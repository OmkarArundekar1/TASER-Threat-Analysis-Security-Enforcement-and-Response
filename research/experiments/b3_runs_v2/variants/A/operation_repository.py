from neo4j_client import (
    create_operation_db,
    update_operation_activity,
    close_operation_db,
    attach_campaign_to_operation,
    expire_stale_operations_db,
    get_campaign_context_data,
    get_operation_context_data,
    get_active_operations,
)


class OperationRepository:

    def create_operation(self):
        return create_operation_db()

    def update_operation(self, operation_id):
        update_operation_activity(operation_id)

    def close_operation(self, operation_id):
        close_operation_db(operation_id)

    def attach_campaign(self, operation_id, campaign_id):
        attach_campaign_to_operation(
            operation_id,
            campaign_id
        )

    def expire_operations(self, timeout):
        return expire_stale_operations_db(timeout)

    def get_campaign(self, campaign_id):
        return get_campaign_context_data(campaign_id)

    def get_operation(self, operation_id):
        return get_operation_context_data(operation_id)

    def get_active_operations(self):
        return get_active_operations()