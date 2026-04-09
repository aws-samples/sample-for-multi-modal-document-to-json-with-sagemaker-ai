function route_request() {
  local path=$(echo $1 | jq -r '.path')
  local method=$(echo $1 | jq -r '.httpMethod')
  local body=$(echo $1 | jq -r '.body')

  case "$path" in
    "/v1/chat/completions")

      echo "$body" | curl -s -X POST "http://localhost:9002/v1/chat/completions" \
           -H "Content-Type: application/json" \
           -d @-
      # curl -s -X POST "http://localhost:9002/v1/chat/completions" \
      #   -H "Content-Type: application/json" \
      #   -d "$body"
      ;;
    "/v1/completions")
      curl -s -X POST "http://localhost:9002/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$body"
      ;;
    "/completion")
      echo "$body" | curl -s -X POST "http://localhost:9002/completion" \
        -H "Content-Type: application/json" \
        -d @-
      ;;
    "/v1/models")
      curl -s -X GET "http://localhost:9002/v1/models" \
        -H "Content-Type: application/json" 
      ;;
    "/health")
      curl -s "http://localhost:9002/health"
      ;;
    *)
      echo '{"error":"Invalid endpoint"}'
      return 1
      ;;
  esac
}


