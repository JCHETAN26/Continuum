import { z } from 'zod';

export const configSchema = z.object({
  // Runtime
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']).default('info'),

  // Postgres (pgvector)
  POSTGRES_USER: z.string().min(1),
  POSTGRES_PASSWORD: z.string().min(1),
  POSTGRES_DB: z.string().min(1),
  POSTGRES_PORT: z.coerce.number().int().positive(),
  DATABASE_URL: z.string().url(),
  SHADOW_DATABASE_URL: z.string().url().optional(),

  // Kafka (Redpanda)
  KAFKA_BROKERS: z.string().min(1),
  KAFKA_CLIENT_ID: z.string().min(1),
  REDPANDA_KAFKA_PORT: z.coerce.number().int().positive(),
  REDPANDA_ADMIN_PORT: z.coerce.number().int().positive(),
  REDPANDA_CONSOLE_PORT: z.coerce.number().int().positive(),

  // Redis
  REDIS_URL: z.string().url(),
  REDIS_PORT: z.coerce.number().int().positive(),

  // Object Storage (MinIO)
  MINIO_ROOT_USER: z.string().min(1),
  MINIO_ROOT_PASSWORD: z.string().min(1),
  MINIO_API_PORT: z.coerce.number().int().positive(),
  MINIO_CONSOLE_PORT: z.coerce.number().int().positive(),

  S3_ENDPOINT: z.string().url(),
  S3_REGION: z.string().min(1),
  S3_ACCESS_KEY_ID: z.string().min(1),
  S3_SECRET_ACCESS_KEY: z.string().min(1),
  S3_BUCKET_DOCUMENTS: z.string().min(1),
  S3_BUCKET_MODELS: z.string().min(1),
  S3_FORCE_PATH_STYLE: z.coerce.boolean(),

  // Embedding model
  EMBEDDING_MODEL: z.string().min(1),
  EMBEDDING_DIM: z.coerce.number().int().positive(),

  // Drift detection
  DRIFT_THRESHOLD: z.coerce.number().min(0).max(1),

  // Observability
  OTEL_SERVICE_NAME: z.string().min(1),
  OTEL_EXPORTER_OTLP_ENDPOINT: z.string().url().optional().or(z.literal('')),
});

export type Config = z.infer<typeof configSchema>;

/**
 * Validates the current process.env against the config schema.
 * Throws an error immediately if the environment is invalid.
 */
export function validateConfig(env: Record<string, string | undefined> = process.env): Config {
  const parsed = configSchema.safeParse(env);

  if (!parsed.success) {
    console.error('❌ Invalid environment configuration:', parsed.error.format());
    process.exit(1);
  }

  return parsed.data;
}

// Export a singleton validated config instance for convenience
export const config = validateConfig();
