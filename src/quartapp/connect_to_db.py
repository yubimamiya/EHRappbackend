import asyncpg
import os
import asyncio

async def connect_to_db():
    try:
        # Fetch DB connection details from environment with exact casing and dashes
        curr_host = os.environ["DB_HOST"]
        curr_port = os.environ.get("DB_PORT", "5432")
        curr_dbname = os.environ["DB_NAME"]
        curr_user = os.environ["DB_USER"]
        curr_password = os.environ["DB_PASSWORD"]

        '''
        curr_host = "c-rec-ex-app-db-cluster.h2jfiotjltxxps.postgres.cosmos.azure.com"
        curr_port = "5432"
        curr_dbname = "citus"
        curr_user = "citus" # YUBI: try using different username
        curr_password = "xenon2025!"
        '''

        # Create asyncpg connection pool (recommended)
        pool = await asyncpg.create_pool(
            host=curr_host,
            port=curr_port,
            database=curr_dbname,
            user=curr_user,
            password=curr_password,
            ssl="require",  # Azure Cosmos DB requires SSL
            min_size=1,
            max_size=5,
        )

        print("Connection to Azure Cosmos DB for PostgreSQL successful.")
        return pool

    except Exception as e:
        print("Failed to connect to the database.")
        print("Error:", e)
        return None

# This runs only when this file is run directly
if __name__ == "__main__":
    async def test():
        pool = await connect_to_db()
        if pool:
            async with pool.acquire() as conn:
                version_row = await conn.fetchrow("SELECT version();")
                print("PostgreSQL version:", version_row["version"])

            await pool.close()

    asyncio.run(test())