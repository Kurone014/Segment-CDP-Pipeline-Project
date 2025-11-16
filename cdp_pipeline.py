# cdp_pipeline.py
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import json
import hashlib
import random
import threading
import time

app = Flask(__name__)

# In-memory store (Redis replacement for Windows testing)
class InMemoryStore:
    def __init__(self):
        self.data = defaultdict(dict)
        self.sets = defaultdict(set)
        self.sorted_sets = defaultdict(list)
        self.locks = {}
    
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
            # Remove old entry if exists
            self.sorted_sets[key] = [(v, s) for v, s in self.sorted_sets[key] if v != value]
            # Add new entry
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
            'share': 8,
            'add_to_cart': 15,
            'checkout_start': 18
        }
        
        # Neuropsychological patterns
        self.hesitation_signals = {
            'cart_abandonment': -10,
            'form_abandon': -5,
            'rapid_back_navigation': -3,
            'price_check_loops': -7
        }
    
    def create_identity_graph(self, event):
        """Advanced identity stitching with cross-device tracking"""
        anonymous_id = event.get('anonymousId')
        user_id = event.get('userId')
        email = event.get('traits', {}).get('email')
        
        # Create identity key based on available identifiers
        if user_id:
            master_id = f"user:{user_id}"
        elif email:
            master_id = f"email:{hashlib.md5(email.encode()).hexdigest()}"
        else:
            master_id = f"anon:{anonymous_id}"
        
        # Store identity mappings
        if user_id and anonymous_id:
            store.sadd(f"identity:{user_id}:anonymous", anonymous_id)
            store.set(f"anonymous:{anonymous_id}:user", user_id)
        
        if email:
            store.set(f"email:{email}:master", master_id)
        
        # Merge behavioral history
        self._merge_identities(master_id, anonymous_id, user_id, email)
        
        return master_id
    
    def _merge_identities(self, master_id, anonymous_id, user_id, email):
        """Merge all behavioral data across identities"""
        # Get all related identities
        related_ids = set()
        if anonymous_id:
            related_ids.add(f"anon:{anonymous_id}")
        if user_id:
            # Get all anonymous IDs for this user
            anon_ids = store.smembers(f"identity:{user_id}:anonymous")
            for aid in anon_ids:
                related_ids.add(f"anon:{aid}")
        
        # Merge histories
        merged_history = []
        for rid in related_ids:
            history = store.zrange(f"history:{rid}", 0, float('inf'), byscore=True)
            merged_history.extend(history)
        
        # Store merged history under master ID
        for event_data in merged_history:
            if isinstance(event_data, str):
                event_json = json.loads(event_data)
                timestamp = event_json.get('timestamp', time.time())
                store.zadd(f"history:{master_id}", {event_data: timestamp})
    
    def calculate_intent_score(self, user_id, event):
        """Advanced psychological scoring with pattern recognition"""
        event_type = event.get('event')
        properties = event.get('properties', {})
        
        # Get user's recent behavior (last 30 min)
        current_time = time.time()
        history_key = f"history:{user_id}"
        recent_events = store.zrange(history_key, current_time - 1800, current_time, byscore=True)
        
        # Parse events
        parsed_events = []
        for e in recent_events:
            try:
                parsed_events.append(json.loads(e))
            except:
                continue
        
        # Calculate base score
        base_score = self.scoring_weights.get(event_type, 1)
        
        # Calculate momentum (acceleration of activity)
        momentum = self._calculate_momentum(parsed_events)
        
        # Detect micro-conversions
        micro_conversions = self._detect_micro_conversions(parsed_events)
        
        # Detect hesitation patterns
        hesitation_score = self._detect_hesitation(parsed_events, event)
        
        # Calculate final intent score
        intent_score = (base_score * momentum) + (micro_conversions * 5) - abs(hesitation_score)
        
        # Detect behavioral patterns
        patterns = self._detect_behavioral_patterns(parsed_events)
        
        # Calculate engagement depth
        engagement_depth = self._calculate_engagement_depth(parsed_events)
        
        return {
            'intent_score': max(0, intent_score),
            'momentum': momentum,
            'patterns': patterns,
            'micro_conversions': micro_conversions,
            'hesitation_score': hesitation_score,
            'engagement_depth': engagement_depth,
            'segment': self._assign_segment(intent_score, patterns, hesitation_score),
            'next_best_action': self._predict_next_action(parsed_events, patterns)
        }
    
    def _calculate_momentum(self, events):
        """Calculate activity acceleration"""
        if len(events) < 2:
            return 1
        
        # Group events by 5-minute windows
        windows = defaultdict(int)
        current_time = time.time()
        
        for event in events:
            event_time = event.get('timestamp', current_time)
            window = int((current_time - event_time) / 300)  # 5-min windows
            windows[window] += 1
        
        # Calculate acceleration
        if len(windows) >= 2:
            recent = windows.get(0, 0)
            previous = windows.get(1, 1)
            momentum = recent / max(previous, 1)
            return min(momentum, 5)  # Cap at 5x
        
        return 1
    
    def _detect_micro_conversions(self, events):
        """Identify small positive actions indicating intent"""
        micro_conversions = 0
        
        for event in events:
            event_type = event.get('event')
            props = event.get('properties', {})
            
            # Content engagement depth
            if event_type == 'page_view' and props.get('time_on_page', 0) > 30:
                micro_conversions += 1
            
            # Scroll depth
            if props.get('scroll_percentage', 0) > 75:
                micro_conversions += 1
            
            # Repeat visits to key pages
            if event_type == 'page_view' and 'pricing' in props.get('page', '').lower():
                micro_conversions += 2
            
            # FAQ or documentation views
            if 'docs' in event_type.lower() or 'faq' in event_type.lower():
                micro_conversions += 1
        
        return micro_conversions
    
    def _detect_hesitation(self, events, current_event):
        """Detect psychological hesitation patterns"""
        hesitation = 0
        
        # Look for abandonment patterns
        cart_adds = sum(1 for e in events if e.get('event') == 'add_to_cart')
        purchases = sum(1 for e in events if e.get('event') == 'purchase')
        
        if cart_adds > 0 and purchases == 0:
            hesitation += self.hesitation_signals['cart_abandonment']
        
        # Form abandonment
        form_starts = sum(1 for e in events if 'form_start' in e.get('event', ''))
        form_completes = sum(1 for e in events if 'form_complete' in e.get('event', ''))
        
        if form_starts > form_completes:
            hesitation += self.hesitation_signals['form_abandon']
        
        # Rapid back navigation (bouncing)
        page_views = [e for e in events if e.get('event') == 'page_view']
        if len(page_views) > 5:
            unique_pages = len(set(e.get('properties', {}).get('page', '') for e in page_views))
            if unique_pages < len(page_views) / 2:
                hesitation += self.hesitation_signals['rapid_back_navigation']
        
        return hesitation
    
    def _detect_behavioral_patterns(self, events):
        """Identify complex behavioral patterns"""
        patterns = []
        
        if len(events) > 10:
            patterns.append('highly_engaged')
        elif len(events) > 5:
            patterns.append('engaged')
        
        # Research behavior
        page_views = [e for e in events if e.get('event') == 'page_view']
        actions = [e for e in events if 'click' in e.get('event', '') or 'form' in e.get('event', '')]
        
        if page_views and actions:
            if len(page_views) > len(actions) * 2:
                patterns.append('researcher')
            elif len(actions) > len(page_views):
                patterns.append('action_oriented')
        
        # Price sensitivity
        pricing_views = [e for e in events if 'pricing' in str(e.get('properties', {})).lower()]
        if len(pricing_views) > 2:
            patterns.append('price_sensitive')
        
        # Technical evaluator
        technical_pages = [e for e in events if any(term in str(e.get('properties', {})).lower() 
                          for term in ['docs', 'api', 'technical', 'integration'])]
        if len(technical_pages) > 3:
            patterns.append('technical_evaluator')
        
        # Comparison shopper
        if self._is_comparison_shopping(events):
            patterns.append('comparison_shopper')
        
        return patterns
    
    def _is_comparison_shopping(self, events):
        """Detect comparison shopping behavior"""
        pages = [e.get('properties', {}).get('page', '') for e in events 
                if e.get('event') == 'page_view']
        
        comparison_keywords = ['vs', 'versus', 'compare', 'alternative', 'competitor']
        comparison_count = sum(1 for page in pages 
                              if any(keyword in page.lower() for keyword in comparison_keywords))
        
        return comparison_count >= 2
    
    def _calculate_engagement_depth(self, events):
        """Calculate how deeply engaged the user is"""
        depth_score = 0
        
        # Unique pages visited
        unique_pages = len(set(e.get('properties', {}).get('page', '') 
                             for e in events if e.get('event') == 'page_view'))
        depth_score += unique_pages * 2
        
        # Different event types
        unique_events = len(set(e.get('event') for e in events))
        depth_score += unique_events * 3
        
        # Time spread
        if events:
            timestamps = [e.get('timestamp', 0) for e in events]
            if timestamps:
                time_spread = max(timestamps) - min(timestamps)
                if time_spread > 300:  # More than 5 minutes
                    depth_score += 10
        
        return depth_score
    
    def _assign_segment(self, score, patterns, hesitation):
        """Dynamic segmentation based on behavioral signals"""
        if score > 50 and 'action_oriented' in patterns and hesitation > -5:
            return 'hot_lead'
        elif score > 30 and 'technical_evaluator' in patterns:
            return 'technical_buyer'
        elif score > 30 and 'researcher' in patterns:
            return 'evaluating'
        elif 'comparison_shopper' in patterns:
            return 'comparing_options'
        elif hesitation < -10:
            return 'hesitant'
        elif score > 20:
            return 'interested'
        else:
            return 'browsing'
    
    def _predict_next_action(self, events, patterns):
        """Predict next best action based on patterns"""
        if 'hot_lead' in str(patterns):
            return 'trigger_sales_outreach'
        elif 'technical_evaluator' in patterns:
            return 'send_technical_docs'
        elif 'price_sensitive' in patterns:
            return 'show_roi_calculator'
        elif 'comparison_shopper' in patterns:
            return 'show_comparison_guide'
        elif 'researcher' in patterns:
            return 'send_educational_content'
        else:
            return 'continue_monitoring'
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
# Continue in next message...