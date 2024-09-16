from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
from .langchain_qa import customer_support_qa

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def query_ai(request):
    try:
        data = json.loads(request.body)
        query = data.get('query')

        if not query:
            return JsonResponse({"error": "No query provided"}, status=400)

        response = customer_support_qa.get_response(query)
        return JsonResponse({"response": response})

    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JsonResponse({"error": "An internal error occurred"}, status=500)