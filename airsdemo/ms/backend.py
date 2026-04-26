#!/usr/bin/env python3
"""
Luna Tech - Real Model Security Scanning Backend
Connects to Palo Alto Networks Prisma AIRS for actual model scanning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)

# Load environment variables from .env file
# Check multiple locations for .env file
env_paths = [
    '/var/www/airsdemo/.env',  # Production path
    os.path.join(os.path.dirname(__file__), '../.env'),  # Parent directory
    os.path.join(os.path.dirname(__file__), '.env'),  # Same directory
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from: {env_path}")
        break

app = Flask (__name__)
CORS (app)  # Enable CORS for frontend communication

# Security Group UUIDs for different source types
SECURITY_GROUPS = {
    'huggingface': 'e3adbe4f-4481-4bd9-9c7c-33b5123d40bf',
    'local': 'd9778b1c-cf1b-493d-a1ca-ef2a99d2bae4',
    'azure': 'd2f3939f-931e-43b3-a47f-df931d4fdc44',
    's3': 'a165a2a2-e864-4426-883d-fe623646a014',
    'gcs': '8c38ffc5-6bd5-4f28-8845-78debfed6b3c'
}

# Model source mapping
MODEL_SOURCES = {
    'distilbert/distilbert-base-uncased': 'huggingface',
    'facebook/opt-125m': 'huggingface',
    'google/flan-t5-small': 'huggingface',
    'microsoft/DialoGPT-medium': 'huggingface',
    'microsoft/phi-2': 'huggingface',
    'meta-llama/Llama-2-7b-chat-hf': 'huggingface',
    'google-bert/bert-base-uncased': 'huggingface'
}


def get_model_security_client ():
    """Initialize the Palo Alto Networks Model Security API Client"""
    try:
        from model_security_client.api import ModelSecurityAPIClient

        # Initialize with your AIRS API endpoint
        client = ModelSecurityAPIClient (
            base_url="https://api.sase.paloaltonetworks.com/aims"
        )
        return client
    except ImportError:
        logger.error ("model_security_client not installed. Run: pip install model-security-client")
        return None
    except Exception as e:
        logger.error (f"Failed to initialize API client: {e}")
        return None


@app.route ('/health', methods=['GET'])
def health_check ():
    """Health check endpoint"""
    return jsonify ({
        'status': 'healthy',
        'timestamp': datetime.utcnow ().isoformat (),
        'service': 'Luna Tech Model Security Scanner'
    })


@app.route ('/api/scan', methods=['POST'])
def scan_model ():
    """
    Scan a model using Palo Alto Networks AIRS

    Expected JSON payload:
    {
        "model_id": "microsoft/DialoGPT-medium",
        "source_type": "huggingface",  # or "local", "azure", "s3", "gcs"
        "model_path": null  # Optional: for local models
    }
    """
    try:
        data = request.get_json ()

        if not data:
            return jsonify ({'error': 'No data provided'}), 400

        model_id = data.get ('model_id')
        source_type = data.get ('source_type', 'huggingface')
        model_path = data.get ('model_path')

        if not model_id:
            return jsonify ({'error': 'model_id is required'}), 400

        # Get the appropriate security group UUID
        security_group_uuid = SECURITY_GROUPS.get (source_type)

        if not security_group_uuid:
            return jsonify ({'error': f'Invalid source_type: {source_type}'}), 400

        logger.info (f"Starting scan for model: {model_id}, source: {source_type}")

        # Initialize API client
        client = get_model_security_client ()

        if not client:
            return jsonify ({
                'error': 'API client initialization failed',
                'details': 'Install model-security-client: pip install model-security-client'
            }), 500

        # Perform the actual scan
        scan_params = {
            'security_group_uuid': security_group_uuid,
            'poll_interval_secs': 5,
            'poll_timeout_secs': 900
        }

        # Get model_uri from request if provided (for cloud storage)
        model_uri = data.get ('model_uri')

        # Choose scan method based on source type
        if source_type == 'huggingface':
            # Scan HuggingFace model
            if not model_uri:
                model_uri = f"https://huggingface.co/{model_id}"
            scan_params['model_uri'] = model_uri
            logger.info (f"Scanning HuggingFace model: {model_uri}")

        elif source_type == 'local' and model_path:
            # Scan local model
            scan_params['model_path'] = model_path
            logger.info (f"Scanning local model: {model_path}")

        elif source_type == 's3' and model_uri:
            # Scan S3 model - must download first per AIRS API requirements
            # API requires BOTH model_path (local) and model_uri (S3) for object storage
            import tempfile
            import shutil

            temp_dir = None
            try:
                import boto3
                from botocore.exceptions import ClientError, NoCredentialsError

                # Parse S3 URI (s3://bucket/path)
                s3_parts = model_uri.replace('s3://', '').split('/', 1)
                bucket_name = s3_parts[0]
                prefix = s3_parts[1] if len(s3_parts) > 1 else ''

                # Create temp directory
                temp_dir = tempfile.mkdtemp(prefix='s3_model_scan_')
                logger.info(f"Downloading S3 model to temp: {temp_dir}")
                logger.info(f"S3 URI: {model_uri} (bucket: {bucket_name}, prefix: {prefix})")

                # Initialize S3 client
                s3_client = boto3.client('s3')

                # List and download all objects with the prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

                file_count = 0
                for page in pages:
                    if 'Contents' not in page:
                        continue

                    for obj in page['Contents']:
                        # Get the object key and create local path
                        s3_key = obj['Key']
                        # Remove the prefix to get relative path
                        relative_path = s3_key[len(prefix):].lstrip('/')
                        if not relative_path:  # Skip if it's the directory itself
                            continue

                        local_file_path = os.path.join(temp_dir, relative_path)

                        # Create directory if needed
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                        # Download file
                        logger.info(f"Downloading: {s3_key}")
                        s3_client.download_file(bucket_name, s3_key, local_file_path)
                        file_count += 1

                logger.info(f"Downloaded {file_count} files from S3")

                if file_count == 0:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    return jsonify({
                        'error': 'S3 download failed',
                        'details': f'No files found at {model_uri}'
                    }), 500

                # For S3 scanning, API requires BOTH model_path and model_uri
                scan_params['model_path'] = temp_dir  # Local path where downloaded
                scan_params['model_uri'] = model_uri  # Original S3 URI
                logger.info(f"Scanning S3 model - Local: {temp_dir}, URI: {model_uri}")

            except (NoCredentialsError, ClientError) as aws_error:
                logger.error(f"AWS error: {aws_error}")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return jsonify({
                    'error': 'S3 access error',
                    'details': str(aws_error)
                }), 500
            except ImportError:
                logger.error("boto3 not installed")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return jsonify({
                    'error': 'boto3 not installed',
                    'details': 'Install with: pip install boto3'
                }), 500
            except Exception as e:
                logger.error(f"S3 download error: {e}")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return jsonify({
                    'error': 'S3 download error',
                    'details': str(e)
                }), 500

        elif source_type in ['azure', 'gcs'] and model_uri:
            # Scan cloud storage model (Azure Blob, or GCS)
            # TODO: Implement download for Azure/GCS similar to S3
            return jsonify({
                'error': 'Not implemented',
                'details': f'{source_type.upper()} scanning requires downloading model first. Not yet implemented.'
            }), 501

        else:
            return jsonify ({
                'error': 'Invalid scan configuration',
                'details': f'For {source_type}: provide model_uri for cloud storage or model_path for local files'
            }), 400

        # Execute the scan
        scan_temp_dir = None  # Track temp dir for S3 cleanup
        if source_type == 's3' and 'model_path' in scan_params:
            scan_temp_dir = scan_params['model_path']

        try:
            result = client.scan (**scan_params)
        except Exception as validation_error:
            # Cleanup S3 temp directory if scan failed
            if scan_temp_dir and os.path.exists(scan_temp_dir):
                import shutil
                try:
                    shutil.rmtree(scan_temp_dir)
                    logger.info(f"Cleaned up S3 temp directory after error: {scan_temp_dir}")
                except:
                    pass

            # Handle validation errors (e.g., invalid model format)
            if "does not follow expected" in str(validation_error):
                return jsonify ({
                    'error': 'Invalid model format',
                    'details': f'HuggingFace models must follow format: author/model-name. Error: {str(validation_error)}'
                }), 400
            raise  # Re-raise other errors

        # Parse and return the results
        # Get eval_outcome and convert enum to string if needed
        eval_outcome_raw = getattr (result, 'eval_outcome', 'UNKNOWN')
        eval_outcome_str = str(eval_outcome_raw).split('.')[-1] if hasattr(eval_outcome_raw, 'name') else str(eval_outcome_raw)

        # Determine if model is allowed or blocked
        is_allowed = eval_outcome_str in ['PASS', 'ALLOWED', 'EvalOutcome.PASS', 'EvalOutcome.ALLOWED']

        scan_result = {
            'scan_id': getattr (result, 'scan_id', 'N/A'),
            'eval_outcome': eval_outcome_str,
            'model_id': model_id,
            'source_type': source_type,
            'security_group_uuid': security_group_uuid,
            'timestamp': datetime.utcnow ().isoformat (),
            'evaluation_time': getattr (result, 'evaluation_time', 'N/A'),
            'rules_checked': getattr (result, 'rules_checked', 0),
            'rules_passed': getattr (result, 'rules_passed', 0),
            'rules_failed': getattr (result, 'rules_failed', 0),
            'findings': [],
            'status': 'ALLOWED' if is_allowed else 'BLOCKED',
            'raw_result': str (result)
        }

        # Extract findings if available
        if hasattr (result, 'findings'):
            scan_result['findings'] = [
                {
                    'severity': getattr (finding, 'severity', 'UNKNOWN'),
                    'rule_name': getattr (finding, 'rule_name', 'Unknown Rule'),
                    'description': getattr (finding, 'description', 'No description'),
                    'remediation': getattr (finding, 'remediation', None)
                }
                for finding in result.findings
            ]

        logger.info (f"Scan completed: {scan_result['eval_outcome']}")

        # Cleanup S3 temp directory after successful scan
        if scan_temp_dir and os.path.exists(scan_temp_dir):
            import shutil
            try:
                shutil.rmtree(scan_temp_dir)
                logger.info(f"Cleaned up S3 temp directory: {scan_temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory {scan_temp_dir}: {cleanup_error}")

        return jsonify ({
            'success': True,
            'scan_result': scan_result
        })

    except Exception as e:
        logger.error (f"Scan failed: {str (e)}", exc_info=True)
        return jsonify ({
            'error': 'Scan failed',
            'details': str (e)
        }), 500


@app.route ('/api/models', methods=['GET'])
def list_models ():
    """List available test models"""
    models = [
        {
            'id': 'distilbert/distilbert-base-uncased',
            'name': 'DistilBERT Base Uncased',
            'source': 'huggingface',
            'description': 'Lightweight, well-vetted model',
            'status': 'safe'
        },
        {
            'id': 'facebook/opt-125m',
            'name': 'Facebook OPT-125M',
            'source': 'huggingface',
            'description': 'Meta OPT small language model',
            'status': 'safe'
        },
        {
            'id': 'google/flan-t5-small',
            'name': 'Google FLAN-T5 Small',
            'source': 'huggingface',
            'description': 'Well-vetted instruction model',
            'status': 'safe'
        },
        {
            'id': 'microsoft/DialoGPT-medium',
            'name': 'DialoGPT Medium',
            'source': 'huggingface',
            'description': 'Conversational AI model',
            'status': 'test'
        }
    ]

    return jsonify ({
        'models': models,
        'security_groups': SECURITY_GROUPS
    })


@app.route ('/api/scan/status/<scan_id>', methods=['GET'])
def get_scan_status (scan_id):
    """Get the status of a running scan"""
    # This would query the AIRS API for scan status
    # For now, return a placeholder
    return jsonify ({
        'scan_id': scan_id,
        'status': 'COMPLETED',
        'message': 'Scan status endpoint - implement with AIRS API query'
    })


@app.route ('/api/download', methods=['POST'])
def download_model ():
    """
    Download a HuggingFace model to the local server

    Expected JSON payload:
    {
        "model_name": "google/flan-t5-small"
    }
    """
    import subprocess
    import shutil

    try:
        data = request.get_json ()

        if not data:
            return jsonify ({'error': 'No data provided'}), 400

        model_name = data.get ('model_name')

        if not model_name:
            return jsonify ({'error': 'model_name is required'}), 400

        # Validate model name format (should be author/model-name)
        if '/' not in model_name:
            return jsonify ({
                'error': 'Invalid model name format',
                'details': 'Model name must follow format: author/model-name (e.g., google/flan-t5-small)'
            }), 400

        # Define download directory
        models_dir = '/var/www/airsdemo/ms-rt/models'

        # Extract model folder name from model_name
        model_folder = model_name.split ('/')[-1]
        model_path = os.path.join (models_dir, model_folder)

        # Check if model already exists
        if os.path.exists (model_path):
            logger.info (f"Model already exists at: {model_path}")
            return jsonify ({
                'success': True,
                'message': 'Model already exists',
                'model_path': model_path,
                'model_name': model_name
            })

        # Ensure models directory exists
        os.makedirs (models_dir, exist_ok=True)

        logger.info (f"Starting download of {model_name} to {model_path}")

        # Construct HuggingFace git URL
        hf_url = f"https://huggingface.co/{model_name}"

        # Check if git-lfs is installed
        lfs_check = subprocess.run (
            ['git', 'lfs', 'version'],
            capture_output=True,
            text=True
        )

        if lfs_check.returncode != 0:
            logger.error ("Git LFS not installed")
            return jsonify ({
                'error': 'Git LFS not installed',
                'details': 'Git LFS is required to download HuggingFace models. Install with: sudo apt-get install git-lfs && git lfs install'
            }), 500

        # Execute git clone with LFS support
        # Note: This may take a while for large models with LFS files
        env = os.environ.copy ()
        env['GIT_LFS_SKIP_SMUDGE'] = '0'  # Ensure LFS files are downloaded

        # Use second disk for temp files (not boot disk)
        # Use models directory which we know has proper permissions
        temp_dir = os.path.join(models_dir, '.tmp')
        os.makedirs (temp_dir, exist_ok=True)
        env['TMPDIR'] = temp_dir
        env['TEMP'] = temp_dir
        env['TMP'] = temp_dir

        result = subprocess.run (
            ['git', 'clone', hf_url, model_path],
            cwd=models_dir,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for large models
            env=env
        )

        if result.returncode != 0:
            logger.error (f"Git clone failed: {result.stderr}")
            # Clean up partial download if it exists
            if os.path.exists (model_path):
                shutil.rmtree (model_path)

            # Check for specific error types
            error_msg = result.stderr

            # Check for disk space issues
            if 'No space left on device' in error_msg or 'unable to write file' in error_msg:
                # Get disk space info
                stat = shutil.disk_usage (models_dir)
                free_gb = stat.free / (1024**3)
                return jsonify ({
                    'error': 'Download failed - Insufficient disk space',
                    'details': f'Not enough disk space available. Free space: {free_gb:.1f}GB. Try a smaller model or free up disk space.'
                }), 500

            # Check if it's a checkout failure (common with LFS or permission issues)
            if 'checkout failed' in error_msg:
                return jsonify ({
                    'error': 'Download failed - Checkout issue',
                    'details': 'Model checkout failed. This is usually due to insufficient disk space or very large model files. Try a smaller model.'
                }), 500

            return jsonify ({
                'error': 'Download failed',
                'details': result.stderr or 'Git clone command failed'
            }), 500

        # Set appropriate permissions (775 so both user and service can access)
        try:
            subprocess.run (
                ['chmod', '-R', '775', model_path],
                check=True
            )
        except subprocess.CalledProcessError:
            logger.warning (f"Could not set permissions on {model_path}")

        logger.info (f"Successfully downloaded {model_name} to {model_path}")

        return jsonify ({
            'success': True,
            'message': 'Model downloaded successfully',
            'model_path': model_path,
            'model_name': model_name
        })

    except subprocess.TimeoutExpired:
        logger.error (f"Download timeout for {model_name}")
        # Clean up partial download
        if os.path.exists (model_path):
            shutil.rmtree (model_path)
        return jsonify ({
            'error': 'Download timeout',
            'details': 'Model download exceeded 30 minute timeout. This may be a very large model.'
        }), 500

    except Exception as e:
        logger.error (f"Download failed: {str (e)}", exc_info=True)
        # Clean up partial download
        if 'model_path' in locals () and os.path.exists (model_path):
            shutil.rmtree (model_path)
        return jsonify ({
            'error': 'Download failed',
            'details': str (e)
        }), 500


if __name__ == '__main__':
    # Check if API client is available
    client = get_model_security_client ()
    if not client:
        logger.warning ("⚠️  Model Security Client not installed!")
        logger.warning ("Install with: pip install model-security-client")
        logger.warning ("Backend will run but scans will fail until installed.")
    else:
        logger.info ("✅ Model Security Client initialized successfully")

    logger.info ("=" * 60)
    logger.info ("🚀 Luna Tech Model Security Scanner Backend")
    logger.info ("=" * 60)
    logger.info (f"Configured Security Groups:")
    for source, uuid in SECURITY_GROUPS.items ():
        logger.info (f"  • {source:12} : {uuid}")
    logger.info ("=" * 60)

    # Run the Flask app
    # Check if running under systemd (disable debug mode for production)
    import sys
    is_systemd = os.environ.get('INVOCATION_ID') is not None

    app.run (
        host='0.0.0.0',
        port=5000,
        debug=not is_systemd  # Disable debug mode when running as systemd service
    )