# cdp_pipeline_fixed.py
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import json
import hashlib
import random
import threading
import time

app = Flask(__name__)

# Enhanced in-memory store
class InMemoryStore:
    def __init__(self):
        self.data = defaultdict(dict)
        self.sets = defaultdict(set)
        self.sorted_sets = defaultdict(list)
        self.locks = {}
        # Add persistent user profiles
        self.user_profiles = {}
        self.user_scores = defaultdict(float)
        
    def sadd(self, key, value):
        self.sets[key].add(value)
        return len(self.sets[key])
    
    def smembers(self, key):
        return self.sets[key]
    
    def zrange(self, key, min_score, max_score, byscore=True):
        results = []
        for item, score in self.sorted_sets[key]:
            if min_score <= score <= max_score:
                results.append(item)
        return results
    
    def zadd(self, key, mapping):
        for value, score in mapping.items():
            # Check if already exists
            exists = False
            for i, (v, s) in enumerate(self.sorted_sets[key]):
                if v == value:
                    self.sorted_sets[key][i] = (value, score)
                    exists = True
                    break
            
            if not exists:
                self.sorted_sets[key].append((value, score))
            
            # Keep sorted by score
            self.sorted_sets[key].sort(key=lambda x: x[1])
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value):
        self.data[key] = value
    
    def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

store = InMemoryStore()

class BehavioralCDP:
    def __init__(self):
        self.scoring_weights = {
            'page_view': 1,
            'button_click': 3,
            'form_start': 5,
            'form_complete': 10,
            'purchase': 20,
            'return_visit': 7,
            'email_open': 2,
            'email_click': 8,
            'scroll_depth': 2,
            'video_play': 4,
            'video_complete': 12,
            'download': 6,
            'download_whitepaper': 8,
            'share': 8,
            'add_to_cart': 15,
            'checkout_start': 18,
            'pricing_page': 3,
            'docs_view': 4,
            'api_docs': 5,
            'faq_view': 3,
            'demo_request': 12,
            'technical_specs': 5,
            'integration_guide': 6,
            'competitor_comparison': 7,
            'vs_page': 7,
            'calculator': 9,
            'exit': 0
        }
        
        # Neuropsychological patterns
        self.hesitation_signals = {
            'cart_abandonment': -10,
            'form_abandon': -5,
            'rapid_back_navigation': -3,
            'price_check_loops': -7
        }
    
    def create_identity_graph(self, event):
        """Advanced identity stitching"""
        anonymous_id = event.get('anonymousId')
        user_id = event.get('userId')
        
        # Create unified ID
        if user_id:
            master_id = f"user:{user_id}"
        else:
            master_id = f"anon:{anonymous_id}"
        
        # Initialize user profile if new
        if master_id not in store.user_profiles:
            store.user_profiles[master_id] = {
                'created_at': time.time(),
                'total_events': 0,
                'total_score': 0,
                'events_history': [],
                'segments_history': [],
                'current_session_start': time.time()
            }
        
        return master_id
    
    def calculate_intent_score(self, user_id, event):
        """FIXED: Accumulate scores properly"""
        event_type = event.get('event')
        properties = event.get('properties', {})
        
        # Get or create user profile
        profile = store.user_profiles.get(user_id, {})
        
        # Get ALL historical events for this user
        current_time = time.time()
        history_key = f"history:{user_id}"
        
        # Store current event first
        event_json = json.dumps(event)
        store.zadd(history_key, {event_json: current_time})
        
        # Now get all events including the new one
        all_events = store.sorted_sets.get(history_key, [])
        
        # Parse all events
        parsed_events = []
        for e, timestamp in all_events:
            try:
                parsed_event = json.loads(e)
                parsed_events.append(parsed_event)
            except:
                continue
        
        # Get recent events (last 30 min) for momentum calculation
        recent_events = [e for e in parsed_events 
                        if e.get('timestamp', 0) > current_time - 1800]
        
        # Calculate base score for current event
        base_score = self.scoring_weights.get(event_type, 1)
        
        # ACCUMULATE total score from ALL events
        total_accumulated_score = sum(
            self.scoring_weights.get(e.get('event', ''), 1) 
            for e in parsed_events
        )
        
        # Calculate momentum based on recent activity
        momentum = self._calculate_momentum(recent_events)
        
        # Detect micro-conversions from entire history
        micro_conversions = self._detect_micro_conversions(parsed_events)
        
        # Detect hesitation patterns
        hesitation_score = self._detect_hesitation(parsed_events, event)
        
        # Calculate CUMULATIVE intent score
        intent_score = (total_accumulated_score * momentum) + (micro_conversions * 5) - abs(hesitation_score)
        
        # Detect behavioral patterns from ALL events
        patterns = self._detect_behavioral_patterns(parsed_events)
        
        # Calculate engagement depth
        engagement_depth = self._calculate_engagement_depth(parsed_events)
        
        # Update user profile
        profile['total_events'] = len(parsed_events)
        profile['total_score'] = intent_score
        profile['last_event_time'] = current_time
        profile['events_history'] = [e.get('event') for e in parsed_events[-10:]]  # Last 10 events
        
        # Determine segment based on ACCUMULATED score and patterns
        segment = self._assign_segment(intent_score, patterns, hesitation_score, len(parsed_events))
        
        return {
            'intent_score': max(0, intent_score),
            'momentum': momentum,
            'patterns': patterns,
            'micro_conversions': micro_conversions,
            'hesitation_score': hesitation_score,
            'engagement_depth': engagement_depth,
            'segment': segment,
            'total_events': len(parsed_events),
            'accumulated_score': total_accumulated_score,
            'session_length': current_time - profile.get('current_session_start', current_time)
        }
    
    def _calculate_momentum(self, events):
        """Calculate activity acceleration"""
        if len(events) < 2:
            return 1.0
        
        # Time-based momentum
        if len(events) >= 3:
            # Events in last 5 minutes vs previous 5 minutes
            five_min_ago = time.time() - 300
            ten_min_ago = time.time() - 600
            
            recent = sum(1 for e in events if e.get('timestamp', 0) > five_min_ago)
            previous = sum(1 for e in events if ten_min_ago < e.get('timestamp', 0) <= five_min_ago)
            
            if previous > 0:
                momentum = recent / previous
                return min(max(momentum, 0.5), 3.0)  # Cap between 0.5 and 3
        
        return 1.2 if len(events) > 5 else 1.0
    
    def _detect_micro_conversions(self, events):
        """Identify small positive actions indicating intent"""
        micro_conversions = 0
        
        for event in events:
            event_type = event.get('event', '')
            props = event.get('properties', {})
            
            # High-value pages
            if 'pricing' in str(props).lower():
                micro_conversions += 2
            if 'demo' in event_type.lower():
                micro_conversions += 3
            if 'contact' in str(props).lower():
                micro_conversions += 2
            
            # Engagement indicators
            if 'download' in event_type.lower():
                micro_conversions += 2
            if 'docs' in event_type.lower() or 'api' in event_type.lower():
                micro_conversions += 1
            
            # Deep engagement
            if props.get('scroll_percentage', 0) > 75:
                micro_conversions += 1
            if props.get('time_on_page', 0) > 60:
                micro_conversions += 1
        
        return micro_conversions
    
    def _detect_hesitation(self, events, current_event):
        """Detect psychological hesitation patterns"""
        hesitation = 0
        
        event_types = [e.get('event', '') for e in events]
        
        # Cart abandonment pattern
        if 'add_to_cart' in event_types and 'purchase' not in event_types:
            hesitation += self.hesitation_signals['cart_abandonment']
        
        # Form abandonment
        form_starts = sum(1 for e in event_types if 'form_start' in e)
        form_completes = sum(1 for e in event_types if 'form_complete' in e)
        
        if form_starts > form_completes:
            hesitation += self.hesitation_signals['form_abandon'] * (form_starts - form_completes)
        
        # Price checking loops
        pricing_views = sum(1 for e in events if 'pricing' in str(e.get('properties', {})).lower())
        if pricing_views > 3:
            hesitation += self.hesitation_signals['price_check_loops']
        
        return hesitation
    
    def _detect_behavioral_patterns(self, events):
        """ENHANCED: Better pattern detection"""
        patterns = []
        
        # Engagement level based on event count
        if len(events) > 15:
            patterns.append('highly_engaged')
        elif len(events) > 8:
            patterns.append('engaged')
        elif len(events) > 4:
            patterns.append('exploring')
        
        # Analyze event types
        event_types = [e.get('event', '') for e in events]
        
        # Research behavior
        research_events = ['docs_view', 'faq_view', 'api_docs', 'technical_specs', 'download_whitepaper']
        research_count = sum(1 for e in event_types if e in research_events)
        
        if research_count >= 3:
            patterns.append('researcher')
        
        # Action-oriented behavior
        action_events = ['form_start', 'form_complete', 'add_to_cart', 'checkout_start', 'demo_request']
        action_count = sum(1 for e in event_types if e in action_events)
        
        if action_count >= 2:
            patterns.append('action_oriented')
        
        # Technical evaluator
        technical_events = ['api_docs', 'technical_specs', 'integration_guide', 'docs_view']
        technical_count = sum(1 for e in event_types if e in technical_events)
        
        if technical_count >= 2:
            patterns.append('technical_evaluator')
        
        # Comparison shopper
        comparison_events = ['competitor_comparison', 'vs_page', 'calculator']
        comparison_count = sum(1 for e in event_types if e in comparison_events)
        
        if comparison_count >= 2:
            patterns.append('comparison_shopper')
        
        # Price sensitivity
        pricing_count = sum(1 for e in events if 'pricing' in str(e).lower())
        if pricing_count >= 2:
            patterns.append('price_sensitive')
        
        return patterns
    
    def _calculate_engagement_depth(self, events):
        """Calculate engagement depth score"""
        if not events:
            return 0
        
        depth_score = 0
        
        # Event diversity
        unique_events = len(set(e.get('event', '') for e in events))
        depth_score += unique_events * 3
        
        # Time investment
        if len(events) > 1:
            first_event = min(e.get('timestamp', time.time()) for e in events)
            last_event = max(e.get('timestamp', time.time()) for e in events)
            time_spent = last_event - first_event
            
            if time_spent > 300:  # More than 5 minutes
                depth_score += 10
            if time_spent > 600:  # More than 10 minutes
                depth_score += 10
        
        # High-value actions
        high_value = ['form_complete', 'add_to_cart', 'checkout_start', 'demo_request', 'download']
        high_value_count = sum(1 for e in events if e.get('event', '') in high_value)
        depth_score += high_value_count * 5
        
        return depth_score
    
    def _assign_segment(self, score, patterns, hesitation, event_count):
        """IMPROVED: Better segmentation logic"""
        # Hot lead: High score + action-oriented + sufficient events
        if score > 50 and 'action_oriented' in patterns and event_count >= 5:
            return 'hot_lead'
        
        # Technical buyer: Technical focus + good engagement
        elif score > 25 and 'technical_evaluator' in patterns:
            return 'technical_buyer'
        
        # Evaluating: Research behavior + decent engagement
        elif score > 20 and 'researcher' in patterns:
            return 'evaluating'
        
        # Comparison shopper
        elif 'comparison_shopper' in patterns and score > 15:
            return 'comparing_options'
        
        # Engaged but not ready
        elif score > 30 and 'engaged' in patterns:
            return 'interested'
        
        # Hesitant
        elif hesitation < -10:
            return 'hesitant'
        
        # Still browsing
        elif score > 10:
            return 'warming_up'
        
        # Default
        else:
            return 'browsing'

# ... Continue with the rest of the code (simulation and routes)
# ============ API ROUTES ============

@app.route('/track', methods=['POST'])
def track_event():
    """Main ingestion endpoint for all events"""
    event = request.json
    
    # Add timestamp if not present
    if 'timestamp' not in event:
        event['timestamp'] = time.time()
    
    # Create/update identity graph
    cdp = BehavioralCDP()
    unified_id = cdp.create_identity_graph(event)
    
    # Calculate real-time behavioral scores
    behavioral_data = cdp.calculate_intent_score(unified_id, event)
    
    # Store enriched event
    enriched_event = store_event(unified_id, event, behavioral_data)
    
    # Route to appropriate destinations
    routing_decisions = route_to_destinations(unified_id, enriched_event, behavioral_data)
    
    # Update user profile
    update_user_profile(unified_id, behavioral_data)
    
    return jsonify({
        'success': True,
        'unified_id': unified_id,
        'behavioral_data': behavioral_data,
        'routing': routing_decisions
    })

@app.route('/identify', methods=['POST'])
def identify_user():
    """Explicitly identify a user (login, signup, etc)"""
    data = request.json
    anonymous_id = data.get('anonymousId')
    user_id = data.get('userId')
    traits = data.get('traits', {})
    
    if not user_id:
        return jsonify({'error': 'userId required'}), 400
    
    # Link identities
    if anonymous_id:
        store.sadd(f"identity:{user_id}:anonymous", anonymous_id)
        store.set(f"anonymous:{anonymous_id}:user", user_id)
    
    # Store user traits
    store.set(f"user:{user_id}:traits", json.dumps(traits))
    
    # Merge historical data
    cdp = BehavioralCDP()
    master_id = cdp.create_identity_graph({
        'anonymousId': anonymous_id,
        'userId': user_id,
        'traits': traits
    })
    
    return jsonify({
        'success': True,
        'master_id': master_id,
        'message': 'User identified and histories merged'
    })

@app.route('/profile/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get comprehensive user profile with all behavioral data"""
    profile = {
        'user_id': user_id,
        'identities': {},
        'behavioral_metrics': {},
        'segments': [],
        'recent_events': [],
        'predictions': {}
    }
    
    # Get all linked identities
    anonymous_ids = list(store.smembers(f"identity:{user_id}:anonymous"))
    profile['identities']['anonymous_ids'] = anonymous_ids
    
    # Get stored profile data
    stored_profile = store.get(f"profile:{user_id}")
    if stored_profile:
        profile.update(json.loads(stored_profile))
    
    # Get recent events
    history = store.zrange(f"history:user:{user_id}", 
                          time.time() - 86400,  # Last 24 hours
                          time.time(), 
                          byscore=True)
    
    for event_str in history[-10:]:  # Last 10 events
        try:
            profile['recent_events'].append(json.loads(event_str))
        except:
            pass
    
    return jsonify(profile)

@app.route('/segment/<segment_name>/users', methods=['GET'])
def get_segment_users(segment_name):
    """Get all users in a specific segment"""
    users = list(store.smembers(f"segment:{segment_name}:users"))
    
    return jsonify({
        'segment': segment_name,
        'user_count': len(users),
        'users': users[:100]  # Limit to 100 for display
    })

@app.route('/analytics/funnel', methods=['POST'])
def analyze_funnel():
    """Analyze conversion funnel"""
    data = request.json
    steps = data.get('steps', [])  # List of event names
    
    funnel_data = {}
    
    # This would normally query your data warehouse
    # For now, returning mock data
    for i, step in enumerate(steps):
        funnel_data[step] = {
            'users': 1000 - (i * 200),  # Mock dropout
            'conversion_rate': 1.0 if i == 0 else (1000 - (i * 200)) / 1000
        }
    
    return jsonify(funnel_data)

# ============ HELPER FUNCTIONS ============

def store_event(user_id, event, behavioral_data):
    """Store event with enrichment in our data warehouse"""
    enriched_event = event.copy()
    enriched_event['behavioral_score'] = behavioral_data['intent_score']
    enriched_event['segment'] = behavioral_data['segment']
    enriched_event['patterns'] = behavioral_data['patterns']
    enriched_event['enriched_at'] = datetime.now().isoformat()
    
    # Store in time-series
    event_json = json.dumps(enriched_event)
    store.zadd(f"history:{user_id}", {event_json: event['timestamp']})
    
    # Also store by event type for analytics
    store.zadd(f"events:{event['event']}", {event_json: event['timestamp']})
    
    # Update segment membership
    store.sadd(f"segment:{behavioral_data['segment']}:users", user_id)
    
    return enriched_event

def route_to_destinations(user_id, event, behavioral_data):
    """Smart routing engine based on behavior"""
    routing = []
    segment = behavioral_data['segment']
    
    if segment == 'hot_lead':
        routing.append({
            'destination': 'salesforce',
            'action': 'create_hot_lead',
            'priority': 'high'
        })
        routing.append({
            'destination': 'slack',
            'action': 'notify_sales_team',
            'message': f"🔥 Hot lead detected: {user_id}"
        })
        
    elif segment == 'technical_buyer':
        routing.append({
            'destination': 'hubspot',
            'action': 'trigger_technical_nurture',
            'campaign': 'technical_decision_maker'
        })
        
    elif segment == 'hesitant':
        routing.append({
            'destination': 'intercom',
            'action': 'trigger_chat_widget',
            'message': 'Need help with your decision?'
        })
        routing.append({
            'destination': 'email',
            'action': 'send_abandonment_email',
            'delay': '1_hour'
        })
    
    elif 'comparison_shopper' in behavioral_data['patterns']:
        routing.append({
            'destination': 'email',
            'action': 'send_comparison_guide',
            'priority': 'medium'
        })
    
    # Log routing decisions
    store.set(f"routing:{user_id}:{time.time()}", json.dumps(routing))
    
    return routing

def update_user_profile(user_id, behavioral_data):
    """Update persistent user profile with latest behavioral data"""
    profile = store.get(f"profile:{user_id}")
    
    if profile:
        profile = json.loads(profile)
    else:
        profile = {
            'created_at': datetime.now().isoformat(),
            'scores': [],
            'segments': []
        }
    
    # Update profile
    profile['last_seen'] = datetime.now().isoformat()
    profile['current_segment'] = behavioral_data['segment']
    profile['current_score'] = behavioral_data['intent_score']
    profile['patterns'] = behavioral_data['patterns']
    
    # Keep history
    profile['scores'].append({
        'score': behavioral_data['intent_score'],
        'timestamp': time.time()
    })
    
    # Store updated profile
    store.set(f"profile:{user_id}", json.dumps(profile))

# ============ SIMULATION ENGINE ============

class UserSimulator:
    """Simulate different user personas for testing"""
    
    def __init__(self):
        self.personas = {
            'hot_buyer': {
                'events': ['page_view', 'page_view', 'pricing_page', 'add_to_cart', 
                          'form_start', 'form_complete', 'checkout_start'],
                'timing': 'fast'  # Quick succession
            },
            'researcher': {
                'events': ['page_view', 'page_view', 'page_view', 'docs_view', 
                          'faq_view', 'page_view', 'download_whitepaper'],
                'timing': 'slow'
            },
            'tire_kicker': {
                'events': ['page_view', 'pricing_page', 'page_view', 'exit'],
                'timing': 'fast'
            },
            'technical_evaluator': {
                'events': ['page_view', 'docs_view', 'api_docs', 'integration_guide',
                          'technical_specs', 'form_start', 'demo_request'],
                'timing': 'medium'
            },
            'comparison_shopper': {
                'events': ['page_view', 'pricing_page', 'competitor_comparison',
                          'vs_page', 'pricing_page', 'calculator'],
                'timing': 'medium'
            }
        }
    
    def simulate_user(self, persona_type, user_id=None):
        """Simulate a complete user journey"""
        if user_id is None:
            user_id = f"sim_{persona_type}_{random.randint(1000, 9999)}"
        
        persona = self.personas.get(persona_type, self.personas['tire_kicker'])
        anonymous_id = f"anon_{random.randint(10000, 99999)}"
        
        events_sent = []
        
        for i, event_type in enumerate(persona['events']):
            # Build event
            event = self._build_event(event_type, anonymous_id, user_id if i > 3 else None)
            
            # Send event
            response = self._send_event(event)
            events_sent.append(response)
            
            # Wait based on timing
            if persona['timing'] == 'fast':
                time.sleep(0.5)
            elif persona['timing'] == 'medium':
                time.sleep(2)
            else:
                time.sleep(5)
        
        return events_sent
    
    def _build_event(self, event_type, anonymous_id, user_id=None):
        """Build a realistic event"""
        base_event = {
            'event': event_type,
            'anonymousId': anonymous_id,
            'timestamp': time.time(),
            'properties': {}
        }
        
        if user_id:
            base_event['userId'] = user_id
        
        # Add properties based on event type
        if event_type == 'page_view':
            base_event['properties'] = {
                'page': random.choice(['/home', '/features', '/about', '/blog']),
                'referrer': 'google.com',
                'time_on_page': random.randint(10, 120)
            }
        elif event_type == 'pricing_page':
            base_event['event'] = 'page_view'
            base_event['properties'] = {
                'page': '/pricing',
                'scroll_percentage': random.randint(30, 100)
            }
        elif event_type == 'add_to_cart':
            base_event['properties'] = {
                'product_id': 'pro_plan',
                'price': 99.99
            }
        
        return base_event
    
    def _send_event(self, event):
        """Send event to CDP"""
        # For testing without running server
        cdp = BehavioralCDP()
        unified_id = cdp.create_identity_graph(event)
        behavioral_data = cdp.calculate_intent_score(unified_id, event)
        
        return {
            'event': event['event'],
            'unified_id': unified_id,
            'segment': behavioral_data['segment'],
            'score': behavioral_data['intent_score']
        }

def run_simulation():
    """Run a complete simulation"""
    simulator = UserSimulator()
    
    print("\n🚀 CDP SIMULATION STARTING...\n")
    print("-" * 50)
    
    personas_to_test = ['hot_buyer', 'researcher', 'technical_evaluator', 
                       'comparison_shopper', 'tire_kicker']
    
    for persona in personas_to_test:
        print(f"\n📊 Simulating {persona.upper()}...")
        results = simulator.simulate_user(persona)
        
        # Show journey progression
        for i, result in enumerate(results):
            print(f"  Step {i+1}: {result['event']}")
            print(f"    → Segment: {result['segment']}")
            print(f"    → Score: {result['score']:.2f}")
        
        print(f"  Final Classification: {results[-1]['segment']}")
    
    print("\n" + "-" * 50)
    print("✅ SIMULATION COMPLETE\n")

# ============ MAIN ============

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        # Run simulation mode
        run_simulation()
    else:
        # Run Flask server
        print("\n🚀 CDP Server starting on http://localhost:5000")
        print("\nEndpoints:")
        print("  POST /track - Track events")
        print("  POST /identify - Identify users")
        print("  GET  /profile/<user_id> - Get user profile")
        print("  GET  /segment/<segment>/users - Get segment users")
        print("\nRun 'python cdp_pipeline.py simulate' to run simulation\n")
        
        app.run(debug=True, port=5000)