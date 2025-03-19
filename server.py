

"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

def measure_time(func, arr):
    start_time = time.time()
    result = func(arr)
    end_time = time.time()
    return (*result, round(end_time - start_time, 6))

# Sorting Algorithms

def bubble_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    
    for i in range(n):
        for j in range(n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                steps.append({"type": "swap", "indices": [j, j + 1], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
    
    return arr, steps, swaps, comparisons

def selection_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]

    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            swaps += 1
            steps.append({"type": "swap", "indices": [i, min_index], "array": arr[:], "swaps": swaps, "comparisons": comparisons})

    return arr, steps, swaps, comparisons

def insertion_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            comparisons += 1
            arr[j + 1] = arr[j]  # Shift the element
            swaps += 1
            steps.append({"type": "swap", "indices": [j, j + 1], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
            j -= 1

        arr[j + 1] = key  # Insert key at the correct position
        
        # Only log if the key is moved
        if j + 1 != i:
            steps.append({"type": "insert", "index": j + 1, "value": key, "array": arr[:], "swaps": swaps, "comparisons": comparisons})

    return arr, steps, swaps, comparisons


def quick_sort_helper(arr, low, high, steps, swaps, comparisons):
    if low < high:
        pivot, s, c = partition(arr, low, high, steps, swaps, comparisons)
        swaps += s
        comparisons += c
        s1, c1 = quick_sort_helper(arr, low, pivot - 1, steps, swaps, comparisons)
        s2, c2 = quick_sort_helper(arr, pivot + 1, high, steps, swaps, comparisons)
        return swaps + s1 + s2, comparisons + c1 + c2
    return swaps, comparisons

def partition(arr, low, high, steps, swaps, comparisons):
    pivot = arr[high]
    i = low - 1
    local_swaps, local_comparisons = 0, 0
    for j in range(low, high):
        local_comparisons += 1
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            local_swaps += 1
            steps.append({"type": "swap", "indices": [i, j], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    local_swaps += 1
    steps.append({"type": "swap", "indices": [i + 1, high], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
    return i + 1, local_swaps, local_comparisons

def quick_sort(arr):
    arr = arr[:]
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    swaps, comparisons = quick_sort_helper(arr, 0, len(arr) - 1, steps, swaps, comparisons)
    return arr, steps, swaps, comparisons

def heapify(arr, n, i, steps, swaps, comparisons):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    local_swaps = 0
    local_comparisons = 0
    
    if left < n:
        local_comparisons += 1
        if arr[left] > arr[largest]:
            largest = left
    
    if right < n:
        local_comparisons += 1
        if arr[right] > arr[largest]:
            largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        local_swaps += 1
        steps.append({"type": "swap", "indices": [i, largest], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
        s, c = heapify(arr, n, largest, steps, swaps + local_swaps, comparisons + local_comparisons)
        local_swaps += s
        local_comparisons += c
    
    return local_swaps, local_comparisons

def heap_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    
    for i in range(n // 2 - 1, -1, -1):
        s, c = heapify(arr, n, i, steps, swaps, comparisons)
        swaps += s
        comparisons += c
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        swaps += 1
        steps.append({"type": "swap", "indices": [i, 0], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
        s, c = heapify(arr, i, 0, steps, swaps, comparisons)
        swaps += s
        comparisons += c
    
    return arr, steps, swaps, comparisons

def merge_sort(arr):
    arr = arr[:]
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    
    def merge(arr, L, M, R, steps, swaps, comparisons):
        left = arr[L:M+1]
        right = arr[M+1:R+1]
        
        i = j = 0
        k = L
        local_swaps = 0
        local_comparisons = 0
        
        while i < len(left) and j < len(right):
            local_comparisons += 1
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
                local_swaps += 1
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            k += 1
        
        while i < len(left):
            arr[k] = left[i]
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            i += 1
            k += 1
        
        while j < len(right):
            arr[k] = right[j]
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            j += 1
            k += 1
            
        return local_swaps, local_comparisons
    
    def merge_sort_internal(arr, left, right, steps, swaps, comparisons):
        if left < right:
            mid = (left + right) // 2
            s1, c1 = merge_sort_internal(arr, left, mid, steps, swaps, comparisons)
            swaps += s1
            comparisons += c1
            
            s2, c2 = merge_sort_internal(arr, mid + 1, right, steps, swaps, comparisons)
            swaps += s2
            comparisons += c2
            
            s3, c3 = merge(arr, left, mid, right, steps, swaps, comparisons)
            return swaps + s3, comparisons + c3
        return 0, 0
    
    swaps, comparisons = merge_sort_internal(arr, 0, len(arr) - 1, steps, swaps, comparisons)
    return arr, steps, swaps, comparisons

sorting_algorithms = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "quick_sort": quick_sort,
    "heap_sort": heap_sort,
    "merge_sort": merge_sort,
}

@app.route("/sort", methods=["POST"])
def sort_numbers():
    data = request.json
    array = data.get("array", [])
    algorithm = data.get("algorithm", "bubble_sort")
    
    if not isinstance(array, list) or not all(isinstance(x, (int, float)) for x in array):
        return jsonify({"error": "Array must contain only numbers."}), 400
    
    if algorithm not in sorting_algorithms:
        return jsonify({"error": "Invalid sorting algorithm."}), 400
    
    sorted_array, steps, swaps, comparisons, execution_time = measure_time(sorting_algorithms[algorithm], array)
    
    # Generate algorithm-specific decision tree info for educational purposes
    algorithm_info = {
        "bubble_sort": "Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if needed. The largest elements 'bubble' to the end.",
        "selection_sort": "Selection Sort finds the minimum element and places it at the beginning, then repeats for the remaining array.",
        "insertion_sort": "Insertion Sort builds the sorted array one item at a time by comparing each with the already sorted portion.",
        "quick_sort": "Quick Sort uses a divide-and-conquer strategy with a pivot. Elements smaller than the pivot go to the left, larger to the right.",
        "heap_sort": "Heap Sort converts the array into a heap data structure, then repeatedly extracts the maximum element.",
        "merge_sort": "Merge Sort divides the array into smaller subarrays, sorts them, and then merges them back together."
    }
    
    return jsonify({
        "sorted_array": sorted_array,
        "steps": steps,
        "swaps": swaps,
        "comparisons": comparisons,
        "execution_time": execution_time,
        "decision_tree": f"{algorithm_info.get(algorithm, 'Sorting algorithm')} completed with {swaps} swaps and {comparisons} comparisons in {execution_time} seconds."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

def measure_time(func, arr):
    start_time = time.time()
    result = func(arr)
    end_time = time.time()
    return (*result, round(end_time - start_time, 6))

# Sorting Algorithms

def bubble_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []
    
    for i in range(n):
        for j in range(n - i - 1):
            comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare elements at indices {j} and {j+1}",
                "elements": [j, j+1],
                "values": [arr[j], arr[j+1]],
                "condition": f"{arr[j]} > {arr[j+1]}"
            }
            
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                decision_step["result"] = "True - Swap elements"
                steps.append({"type": "swap", "indices": [j, j + 1], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
            else:
                decision_step["result"] = "False - No swap needed"
                
            decision_steps.append(decision_step)
    
    return arr, steps, swaps, comparisons, decision_steps

def selection_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []

    for i in range(n):
        min_index = i
        decision_steps.append({
            "type": "selection",
            "description": f"Start iteration {i+1}: Find minimum element in positions {i} to {n-1}",
            "currentIndex": i,
            "currentMinIndex": min_index,
            "currentMinValue": arr[min_index]
        })
        
        for j in range(i + 1, n):
            comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare current minimum ({arr[min_index]} at index {min_index}) with element at index {j} ({arr[j]})",
                "elements": [min_index, j],
                "values": [arr[min_index], arr[j]],
                "condition": f"{arr[j]} < {arr[min_index]}"
            }
            
            if arr[j] < arr[min_index]:
                min_index = j
                decision_step["result"] = f"True - Update minimum index to {j}"
                decision_step["newMinIndex"] = j
                decision_step["newMinValue"] = arr[j]
            else:
                decision_step["result"] = "False - Keep current minimum"
                
            decision_steps.append(decision_step)
            
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            swaps += 1
            decision_steps.append({
                "type": "swap",
                "description": f"Swap minimum element ({arr[i]}) to position {i}",
                "elements": [i, min_index],
                "values": [arr[min_index], arr[i]]
            })
            steps.append({"type": "swap", "indices": [i, min_index], "array": arr[:], "swaps": swaps, "comparisons": comparisons})

    return arr, steps, swaps, comparisons, decision_steps

def insertion_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        decision_steps.append({
            "type": "insertion",
            "description": f"Take element {key} at position {i} and find its correct position in the sorted part",
            "key": key,
            "keyIndex": i,
            "sortedPart": arr[:i]
        })
        
        while j >= 0 and arr[j] > key:
            comparisons += 1
            decision_steps.append({
                "type": "comparison",
                "description": f"Compare key ({key}) with element at position {j} ({arr[j]})",
                "elements": [j, "key"],
                "values": [arr[j], key],
                "condition": f"{arr[j]} > {key}",
                "result": "True - Shift element to the right"
            })
            
            arr[j + 1] = arr[j]  # Shift the element
            swaps += 1
            steps.append({"type": "swap", "indices": [j, j + 1], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
            j -= 1

        arr[j + 1] = key  # Insert key at the correct position
        
        # Only log if the key is moved
        if j + 1 != i:
            decision_steps.append({
                "type": "insert",
                "description": f"Insert key {key} at position {j+1}",
                "keyValue": key,
                "insertPosition": j+1,
                "originalPosition": i
            })
            steps.append({"type": "insert", "index": j + 1, "value": key, "array": arr[:], "swaps": swaps, "comparisons": comparisons})

    return arr, steps, swaps, comparisons, decision_steps

def quick_sort(arr):
    arr = arr[:]
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []
    
    def partition(arr, low, high, steps, swaps, comparisons, decision_steps, depth):
        pivot = arr[high]
        i = low - 1
        local_swaps, local_comparisons = 0, 0
        
        decision_steps.append({
            "type": "pivot",
            "description": f"Choose pivot: {pivot} at index {high}",
            "pivot": pivot,
            "pivotIndex": high,
            "partitionRange": [low, high],
            "depth": depth
        })
        
        for j in range(low, high):
            local_comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare element {arr[j]} at index {j} with pivot {pivot}",
                "elements": [j, "pivot"],
                "values": [arr[j], pivot],
                "condition": f"{arr[j]} < {pivot}",
                "depth": depth
            }
            
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                local_swaps += 1
                decision_step["result"] = f"True - Swap elements at indices {i} and {j}"
                steps.append({"type": "swap", "indices": [i, j], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            else:
                decision_step["result"] = "False - No swap needed"
                
            decision_steps.append(decision_step)
            
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        local_swaps += 1
        
        decision_steps.append({
            "type": "pivotPlacement",
            "description": f"Place pivot {pivot} at its correct position {i+1}",
            "elements": [i+1, high],
            "values": [arr[i+1], arr[high]],
            "pivotFinalIndex": i+1,
            "depth": depth
        })
        
        steps.append({"type": "swap", "indices": [i + 1, high], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
        return i + 1, local_swaps, local_comparisons
    
    def quick_sort_helper(arr, low, high, steps, swaps, comparisons, decision_steps, depth=0):
        if low < high:
            decision_steps.append({
                "type": "recursion",
                "description": f"Partition array from index {low} to {high}",
                "range": [low, high],
                "subarray": arr[low:high+1],
                "depth": depth
            })
            
            pivot, s, c = partition(arr, low, high, steps, swaps, comparisons, decision_steps, depth)
            swaps += s
            comparisons += c
            
            decision_steps.append({
                "type": "subproblems",
                "description": f"Split into two subproblems: indices {low} to {pivot-1} and {pivot+1} to {high}",
                "leftSubarray": arr[low:pivot],
                "rightSubarray": arr[pivot+1:high+1],
                "depth": depth
            })
            
            s1, c1 = quick_sort_helper(arr, low, pivot - 1, steps, swaps, comparisons, decision_steps, depth + 1)
            s2, c2 = quick_sort_helper(arr, pivot + 1, high, steps, swaps, comparisons, decision_steps, depth + 1)
            return swaps + s1 + s2, comparisons + c1 + c2
        return swaps, comparisons
    
    swaps, comparisons = quick_sort_helper(arr, 0, len(arr) - 1, steps, swaps, comparisons, decision_steps)
    return arr, steps, swaps, comparisons, decision_steps

def merge_sort(arr):
    arr = arr[:]
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []
    
    def merge(arr, L, M, R, steps, swaps, comparisons, decision_steps, depth):
        left = arr[L:M+1]
        right = arr[M+1:R+1]
        
        decision_steps.append({
            "type": "merge",
            "description": f"Merge subarrays: {left} and {right}",
            "leftSubarray": left,
            "rightSubarray": right,
            "leftRange": [L, M],
            "rightRange": [M+1, R],
            "depth": depth
        })
        
        i = j = 0
        k = L
        local_swaps = 0
        local_comparisons = 0
        
        while i < len(left) and j < len(right):
            local_comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare elements: {left[i]} from left array and {right[j]} from right array",
                "leftValue": left[i],
                "rightValue": right[j],
                "condition": f"{left[i]} <= {right[j]}",
                "depth": depth
            }
            
            if left[i] <= right[j]:
                arr[k] = left[i]
                decision_step["result"] = f"True - Take element {left[i]} from left array"
                i += 1
            else:
                arr[k] = right[j]
                decision_step["result"] = f"False - Take element {right[j]} from right array"
                j += 1
                local_swaps += 1
                
            decision_steps.append(decision_step)
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            k += 1
        
        while i < len(left):
            decision_steps.append({
                "type": "remaining",
                "description": f"Copy remaining element {left[i]} from left array",
                "value": left[i],
                "position": k,
                "depth": depth
            })
            
            arr[k] = left[i]
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            i += 1
            k += 1
        
        while j < len(right):
            decision_steps.append({
                "type": "remaining",
                "description": f"Copy remaining element {right[j]} from right array",
                "value": right[j],
                "position": k,
                "depth": depth
            })
            
            arr[k] = right[j]
            steps.append({"type": "merge", "index": k, "value": arr[k], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            j += 1
            k += 1
            
        return local_swaps, local_comparisons
    
    def merge_sort_internal(arr, left, right, steps, swaps, comparisons, decision_steps, depth=0):
        if left < right:
            mid = (left + right) // 2
            
            decision_steps.append({
                "type": "split",
                "description": f"Split array at index {mid}",
                "range": [left, right],
                "midpoint": mid,
                "leftSubarray": arr[left:mid+1],
                "rightSubarray": arr[mid+1:right+1],
                "depth": depth
            })
            
            s1, c1 = merge_sort_internal(arr, left, mid, steps, swaps, comparisons, decision_steps, depth + 1)
            swaps += s1
            comparisons += c1
            
            s2, c2 = merge_sort_internal(arr, mid + 1, right, steps, swaps, comparisons, decision_steps, depth + 1)
            swaps += s2
            comparisons += c2
            
            s3, c3 = merge(arr, left, mid, right, steps, swaps, comparisons, decision_steps, depth)
            return swaps + s3, comparisons + c3
        return 0, 0
    
    swaps, comparisons = merge_sort_internal(arr, 0, len(arr) - 1, steps, swaps, comparisons, decision_steps)
    return arr, steps, swaps, comparisons, decision_steps

def heap_sort(arr):
    arr = arr[:]
    n = len(arr)
    swaps, comparisons = 0, 0
    steps = [{"array": arr[:], "swaps": swaps, "comparisons": comparisons}]
    decision_steps = []
    
    def heapify(arr, n, i, steps, swaps, comparisons, decision_steps, phase, depth=0):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        local_swaps = 0
        local_comparisons = 0
        
        decision_steps.append({
            "type": "heapify",
            "description": f"Heapify at index {i} with value {arr[i]}",
            "nodeIndex": i,
            "nodeValue": arr[i],
            "phase": phase,
            "depth": depth
        })
        
        if left < n:
            local_comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare parent {arr[i]} with left child {arr[left]} at index {left}",
                "elements": [i, left],
                "values": [arr[i], arr[left]],
                "condition": f"{arr[left]} > {arr[largest]}",
                "phase": phase,
                "depth": depth
            }
            
            if arr[left] > arr[largest]:
                largest = left
                decision_step["result"] = f"True - Update largest to left child (index {left})"
            else:
                decision_step["result"] = "False - Keep current largest"
                
            decision_steps.append(decision_step)
        
        if right < n:
            local_comparisons += 1
            decision_step = {
                "type": "comparison",
                "description": f"Compare current largest {arr[largest]} with right child {arr[right]} at index {right}",
                "elements": [largest, right],
                "values": [arr[largest], arr[right]],
                "condition": f"{arr[right]} > {arr[largest]}",
                "phase": phase,
                "depth": depth
            }
            
            if arr[right] > arr[largest]:
                largest = right
                decision_step["result"] = f"True - Update largest to right child (index {right})"
            else:
                decision_step["result"] = "False - Keep current largest"
                
            decision_steps.append(decision_step)
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            local_swaps += 1
            
            decision_steps.append({
                "type": "swap",
                "description": f"Swap {arr[largest]} at index {i} with {arr[i]} at index {largest}",
                "elements": [i, largest],
                "values": [arr[i], arr[largest]],
                "phase": phase,
                "depth": depth
            })
            
            steps.append({"type": "swap", "indices": [i, largest], "array": arr[:], "swaps": swaps + local_swaps, "comparisons": comparisons + local_comparisons})
            s, c = heapify(arr, n, largest, steps, swaps + local_swaps, comparisons + local_comparisons, decision_steps, phase, depth + 1)
            local_swaps += s
            local_comparisons += c
        
        return local_swaps, local_comparisons
    
    decision_steps.append({
        "type": "phase",
        "description": "Phase 1: Build max heap",
        "array": arr[:]
    })
    
    for i in range(n // 2 - 1, -1, -1):
        s, c = heapify(arr, n, i, steps, swaps, comparisons, decision_steps, "build", 0)
        swaps += s
        comparisons += c
    
    decision_steps.append({
        "type": "phase",
        "description": "Phase 2: Extract elements from heap",
        "array": arr[:]
    })
    
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        swaps += 1
        
        decision_steps.append({
            "type": "extract",
            "description": f"Extract maximum element {arr[i]} and place at position {i}",
            "elements": [0, i],
            "values": [arr[0], arr[i]]
        })
        
        steps.append({"type": "swap", "indices": [i, 0], "array": arr[:], "swaps": swaps, "comparisons": comparisons})
        s, c = heapify(arr, i, 0, steps, swaps, comparisons, decision_steps, "extract", 0)
        swaps += s
        comparisons += c
    
    return arr, steps, swaps, comparisons, decision_steps

sorting_algorithms = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "quick_sort": quick_sort,
    "heap_sort": heap_sort,
    "merge_sort": merge_sort,
}

# Algorithm explanations with more detailed decision tree info
algorithm_explanations = {
    "bubble_sort": {
        "short": "Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if needed. The largest elements 'bubble' to the end.",
        "time_complexity": "O(n²) average and worst case",
        "space_complexity": "O(1)",
        "stability": "Stable",
        "adaptive": "Yes - can be optimized to stop early if no swaps are made in a pass",
        "decision_explanation": "For each element, Bubble Sort decides whether to swap with the next element by comparing their values. If the current element is greater than the next, they are swapped."
    },
    "selection_sort": {
        "short": "Selection Sort finds the minimum element and places it at the beginning, then repeats for the remaining array.",
        "time_complexity": "O(n²) in all cases",
        "space_complexity": "O(1)",
        "stability": "Not stable",
        "adaptive": "No",
        "decision_explanation": "Selection Sort decides which element to place next by finding the minimum value in the unsorted portion and moving it to its correct position."
    },
    "insertion_sort": {
        "short": "Insertion Sort builds the sorted array one item at a time by comparing each with the already sorted portion.",
        "time_complexity": "O(n²) average and worst case, O(n) best case",
        "space_complexity": "O(1)",
        "stability": "Stable",
        "adaptive": "Yes - works well for nearly sorted arrays",
        "decision_explanation": "Insertion Sort decides where to insert each element by comparing it with already sorted elements and shifting larger elements to make space."
    },
    "quick_sort": {
        "short": "Quick Sort uses a divide-and-conquer strategy with a pivot. Elements smaller than the pivot go to the left, larger to the right.",
        "time_complexity": "O(n log n) average case, O(n²) worst case",
        "space_complexity": "O(log n) to O(n)",
        "stability": "Not stable",
        "adaptive": "No",
        "decision_explanation": "Quick Sort decides how to partition the array by selecting a pivot and arranging elements so smaller values are on the left and larger on the right."
    },
    "merge_sort": {
        "short": "Merge Sort divides the array into smaller subarrays, sorts them, and then merges them back together.",
        "time_complexity": "O(n log n) in all cases",
        "space_complexity": "O(n)",
        "stability": "Stable",
        "adaptive": "No",
        "decision_explanation": "Merge Sort decides how to combine sorted subarrays by comparing elements from both and selecting the smaller one first."
    },
    "heap_sort": {
        "short": "Heap Sort converts the array into a heap data structure, then repeatedly extracts the maximum element.",
        "time_complexity": "O(n log n) in all cases",
        "space_complexity": "O(1)",
        "stability": "Not stable",
        "adaptive": "No",
        "decision_explanation": "Heap Sort makes decisions in two phases: first building a max heap, then extracting the maximum element and re-heapifying."
    }
}

@app.route("/sort", methods=["POST"])
def sort_numbers():
    data = request.json
    array = data.get("array", [])
    algorithm = data.get("algorithm", "bubble_sort")
    
    if not isinstance(array, list) or not all(isinstance(x, (int, float)) for x in array):
        return jsonify({"error": "Array must contain only numbers."}), 400
    
    if algorithm not in sorting_algorithms:
        return jsonify({"error": "Invalid sorting algorithm."}), 400
    
    sorted_array, steps, swaps, comparisons, decision_steps, execution_time = measure_time(sorting_algorithms[algorithm], array)
    
    # Get algorithm explanation details
    algorithm_detail = algorithm_explanations.get(algorithm, {})
    
    return jsonify({
        "sorted_array": sorted_array,
        "steps": steps,
        "swaps": swaps,
        "comparisons": comparisons,
        "execution_time": execution_time,
        "algorithm_info": algorithm_detail,
        "decision_steps": decision_steps,
        "decision_tree": f"{algorithm_detail.get('short', 'Sorting algorithm')} completed with {swaps} swaps and {comparisons} comparisons in {execution_time} seconds."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

